# Lauf 5: Deployment des Prefetch-Builds, Stabilitätstest, Compute-Buffer im Allocator (2026-11-25)

Rohdaten `roh5/`. Externe Arbeiten zur Maske: `kq-maske-extern.md`.
Legende: **gemessen** / **gerechnet** (aus gemessenen Teilen) / **behauptet** (fremd, nicht geprüft).

## 1. Setup

- Build: llama.cpp `45b6604b2` + `patch-prefetch-port.diff`, Prefetch an mit `GGML_SCHED_PREFETCH_EXPERTS=3`.
- Zwei Konfigurationen, beide `--load-mode dio`, `-np 1`:
  - `coding-pf`: ub 4096, 128k Kontext, mmproj auf der CPU.
  - `coding-fast`: wie coding-pf, aber ub 8192 und 64k.
- 16k Prefill mit coding-pf: **2345 t/s** (ohne Prefetch ~1400).

## 2. Stabilitätstest (gemessen, eigener llama-server, Argumente wie in Abschnitt 1)

| | coding-pf (ub 4096, 128k) | coding-fast (ub 8192, 64k) |
|---|---|---|
| Laden | 15,3 s | 15,3 s |
| Längen in Zufallsreihenfolge, 500 … 66k | alle ok | alle ok |
| 16k / 32k / 48k Prefill | 2384 / 2280 / 2149 t/s | 3580 / 3276 / 2981 t/s |
| randnah | **130 000 Token ok, 1654 t/s**, tg 27,8 | 64 500 Token ok, 2762 t/s |
| Prefix-Reuse, 30k-Dokument + 4 Turns | Turn 1–4: 27–112 Token neu gerechnet, `cache_n` ~30k | ebenso |
| identischer Request ×3 | 2. und 3.: 4 Token neu, gleicher Text | ebenso |
| Tool-Calling (`model: qwen3.6-35b-a3b`) | `tool_calls` korrekt, Folgeantwort nutzt das Ergebnis | ebenso |
| Bild (mmproj CPU) | „Rot“ | „Rot“ |
| 2 Requests gleichzeitig (np 1) | beide ok, der zweite wartet (7,2 / 14,5 s) | ebenso |
| VRAM-Peak gesamt | **7789 MiB** (~0,4 GB frei) | 7271 MiB |
| RAM nach dem Test | 26,6 GB belegt, 4,9 GB verfügbar | 26,6 / 4,8 GB |
| Fehler im Log | keine | keine |

- Der RAM ist jetzt die knappere Ressource: dio (~22,4 GB) + mmproj (~1 GB) + Prompt-Cache (bis 4 GB).
  Nach dem Test waren noch 4,8 GB verfügbar. Daneben darf nichts Großes laufen.
- Nicht getestet: Dauerlast über Stunden, mehrere Bilder in langem Kontext, 131072 volle Token.

## 3. Compute-Buffer: was am Peak wirklich liegt (gemessen im Allocator)

Methode: `patch-allocdump.diff` auf den pf-Build (nur `ggml-alloc.c`, `GGML_ALLOCATOR_DEBUG` + Dump nach stderr bei
`GGML_ALLOC_DUMP=1`). Bei jedem neuen Maximum je Buffer-Typ schreibt der Patch die live Tensoren raus. Der Dump entsteht beim
Planen vor dem `cudaMalloc`, deshalb sind auch die Fälle drin, die nicht passen. Kontrolle: Der Peak im Dump ist
genau die Größe, an der `cudaMalloc` scheitert (2978,05 und 3842,06 MiB). Auswertung: `auswert5-alloc.py` → `auswertung-lauf5-alloc.txt`.

Peak des pp-Graphen (CUDA0), alles in MiB:

| ub | ctx | Peak | KQ-Maske | `ffn_moe_down` + `ffn_moe_weighted` (f32, 2048×8×ub) | Aktivierungen | Lücken |
|---|---|---|---|---|---|---|
| 2048 | 131072 | 994 | 512 | (Peak liegt anders: Experten-Kopie 210) | 80 | 190 |
| 4096 | 65536 | 1602 | 512 | 256 + 256 | 96 | 482 |
| 4096 | 131072 | 2114 | **1024** | 256 + 256 | 96 | 482 |
| 6144 | 131072 | 2978 ✗ | **1536** | 384 + 384 | 144 | 530 |
| 8192 | 65536 | 2818 | 1024 | 512 + 512 | 192 | 578 |
| 8192 | 131072 | 3842 ✗ | **2048** | 512 + 512 | 192 | 578 |

- **Bestätigt:** Die KQ-Maske ist ein Tensor `attn_inp_kq_mask`, f16, `n_kv × n_tokens`. Er bleibt über den ganzen Graphen live,
  und die Reserve rechnet mit `n_kv` = voller Kontext (`llama-kv-cache.cpp:2957`, `n_kv = kv->get_size()`).
  Bei 128k ist sie der größte Einzelposten (53 % bei ub 8192).
- **Korrektur:** `llama-graph.cpp:1098`, wie in `ubatch-und-prefetch.md` zitiert, ist der DSV4-Pfad. Unser Modell nutzt
  `build_attn_inp_kq_mask` in `llama-graph.cpp:30-47`, mit derselben Form.
- **Peak ohne Maske (gerechnet aus gemessenen Teilen):** ub 8192 → 3842 − 2048 = **1794 MiB**, bei 64k und 128k gleich. Das ist weniger als
  die 2114 MiB, mit denen ub 4096 bei 128k heute läuft. Gerechneter VRAM für ub 8192 + 128k + Prefetch:
  7285 (gemessen, 64k) + 1280 (KV 64k→128k) − 1024 (Maske bei 64k) ≈ **7540 MiB**. Nicht gestartet, nicht gemessen.
- **Zweiter Posten, neu:** `ffn_moe_down` (MUL_MAT_ID) und `ffn_moe_weighted` (MUL mit den Router-Gewichten) liegen am Peak nebeneinander,
  je `n_embd × n_used × n_tokens` f32, bei ub 8192 also 2 × 512 MiB. MUL kann eigentlich inplace laufen (`ggml_op_can_inplace`),
  der Allocator verwendet den Parent hier aber nicht wieder. Warum, ist **nicht geklärt**. Die Bedingungen in
  `ggml_gallocr_allocate_node` habe ich gelesen, aber nicht mit `AT_PRINTF` nachgesehen, welche davon greift. Ginge es inplace,
  wären bei ub 8192 bis zu 512 MiB weniger nötig (gerechnet), und das wäre exakt.
- Die Lücke von 210 MiB zwischen Maske und MoE-Ausgaben ist der Platz der Staging-Kopie eines Expertentensors
  (`CUDA0#blk.N.ffn_down_exps.weight#0`, q6_K, 210 MiB, im ub-2048-Dump direkt zu sehen). Ob sie genau diese Lücke ist, ist aus
  den Offsets geschlossen.

## 4. Prefetch-Puffer außerhalb des Prefills (nur gelesen, nichts geändert)

- Die Slots legt `ggml_backend_sched_prefetch_init` beim ersten Prefill-Split an (3 × größter Expertentensor = 630 MiB, eigene
  `ggml_backend_buffer`). Freigegeben werden sie erst in `ggml_backend_sched_free`. Im Decode benutzt sie niemand,
  weil `ids->ne[0]*ne[1] < 2*n_expert` den Pfad abschaltet.
- **Freigeben nach dem Prefill** wäre einfach (free nach dem letzten Split, neu anlegen beim nächsten Prefill). Im Prozess
  fehlt aber ein Abnehmer: KV und Compute-Buffer sind fest reserviert, nur der CUDA-Pool für Temporaries wächst. Die
  630 MiB kämen anderen Prozessen zugute, und jeder Prefill kostete dann ein `cudaMalloc`, das scheitern kann, wenn
  in der Zwischenzeit jemand anderes das VRAM genommen hat. Gewinn ohne Abnehmer: keiner. Das Risiko ist real.
- **Wiederverwenden ist der bessere Ansatz:** Mit Prefetch landen die Experten in den Slots, die Staging-Kopie
  `input_cpy` im Compute-Buffer (210 MiB je Expertentensor) wird aber trotzdem reserviert. Ließe man die Slots in
  diesen Platz zeigen, oder legte man nur 2 eigene Slots an und nähme die Staging-Kopie als dritten, wären ~210 MiB
  gespart (gerechnet). Dafür muss man die Lebensdauer in gallocr kennen, weil der Platz zwischen Splits wiederverwendet wird.

## 5. Bewertung der Wege zu ub 8192 + 128k

1. **Maske ableiten (Port von stew675 V3, `kq-maske-extern.md`):** spart 2 GiB, ist exakt, Aufwand 2–4 Tage, mit Sicherheit machbar.
   Zwei Haken für uns, beide im Fork-Code gesehen:
   - Mit `--mmproj` und einem Bild im Kontext fällt der Fork für jeden folgenden Batch auf die dichte Maske zurück
     (M-RoPE-2D). Die Reserve müsste sie dann doch vorhalten.
   - Auf NVIDIA fehlt der kausale Tile-Skip, ein kleiner exakter Zusatzkernel ist nötig.

   Upstream gibt es nichts Vergleichbares. PR #29510 steckt in unserem Build, hilft bei `-np 1` aber nicht.
2. **Inplace für `ffn_moe_weighted`:** bis 512 MiB bei ub 8192, vermutlich wenige Zeilen, Ursache noch offen. Billig zu prüfen,
   sollte vor dem Masken-Port kommen.
3. **Staging-Kopie und Prefetch-Slots zusammenlegen:** ~210 MiB.

2 + 3 allein reichen nicht für ub 8192 bei 128k (gerechnet: 3842 − 512 − 210 ≈ 3120 MiB Compute-Buffer, also mehr als heute).
Ohne Bilder im Kontext reicht 1 allein (≈ 1794 MiB). Mit Bildern im Kontext braucht 1 zusätzlich die M-RoPE-Ableitung,
die es im Fork nicht gibt.
