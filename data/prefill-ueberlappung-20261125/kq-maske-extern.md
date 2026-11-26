# KQ-Maske: externe Arbeiten geprüft (2026-11-25)

Codelektüre, nichts gebaut oder ausgeführt. Klone unter `/tmp/kqmask-recherche/`:
`rdna/` (stew675/llama-cpp-rdna-boosts @ `7bfadb0e`), `upstream/` (ggml-org/llama.cpp, blobless, master @ `9c003a97a`, 2026-11-25),
`fork/` = Worktree auf `919794500` + alle 19 Default-Patches per `git am` (Tree `8962b7a40` = `release.json` `.tree`, also exakt der veröffentlichte Stand r361).

Legende: **[Code]** = im Code gesehen, **[Text]** = nur README/PR-Text/Doku, **[fremd gemessen]** = Messung anderer, **[nicht gefunden]**.

## Ausgangslage bei uns (Upstream 45b6604b2)

- [Code] `src/llama-graph.cpp:30-47` (bei 45b6604b2): `attn_inp_kq_mask` = `ggml_new_tensor_4d(F16, n_kv, n_tokens/n_stream, 1, n_stream)`, `n_stream = kv_unified ? 1 : n_seqs_unq`. Ein Masken-Tensor für alle Attention-Layer. Upstream-master hat die Funktion unverändert (nur `force_f16`-Parameter dazu).
- [Code, lokales Log] Modell: `n_layer 40`, `full_attention_interval 4` → 10 Full-Attention-Layer, `head_count 16`, `head_count_kv 2` (GQA 8), `key_length 256` (`bench-data/loadmode/langkontext/1-auto-server.log`).
- Rechnung (nicht gemessen): Maske bei Reserve mit n_kv = 131072: ub 8192 → 131072·8192·2 B = **2048 MiB**; je +2048 ub → **512 MiB** der beobachteten ~864 MiB (≈59 %). Dazu kommt die gleiche Größe noch einmal im CUDA_Host-Compute-Buffer (RAM, kein VRAM).
- [Code] FA-Kernelwahl sm_86, `fattn.cu:562ff` (45b6604b2): D=256 erlaubt, `turing_mma_available(cc)` → für Prefill (`Q->ne[1]` groß) **`BEST_FATTN_KERNEL_MMA_F16`**; VEC nur bei `Q->ne[1]==1` usw. `gqa_opt_applies` verlangt `mask != nullptr` (wichtig für einen Port, s. u.).
- [Code] Kausales Tile-Skipping hängt an der Maske: `fattn-common.cuh` (45b6604b2) ~Z. 1119-1138, `scan_mask = !use_sparse && mask && n_kv_cols % 256 == 0 && (Q->ne[1] >= 1024 || …)` → `flash_attn_mask_to_KV_max` liest die Maske, um pro Q-Tile die oberen KV-Blöcke auszulassen. Bei ub 4096/8192 aktiv.

## 1. stew675/llama-cpp-rdna-boosts, `LLAMA_KQ_MASK_DERIVED`

**Existiert: bestätigt** (`gh api`, nicht Fork, letzter Push 2026-11-25). Patch-Sammlung (19 Default-Blöcke, `git am`) auf upstream **`919794500`** (2026-11-20, ~b12597); `45b6604b2` ist Vorfahre davon, 441 Commits davor. Release `v18-919794500-r361`.
Der Derived-Mask-Code steckt in Block 05 (`patches/0005-…flash-attention-band-native-qua.patch`, 12 291 Zeilen, mit vielem anderem vermischt). Ursprünglich „V3“ aus Block 15 (`archive/work/block-15-campaign-wins/block-15-campaign-wins.patch`, 3 819 Zeilen, 7 Features); grobe Zählung der Hunks mit derived/cell_pos/tok_lo: ~1 200 `+`-Zeilen, davon ~400 qwen4exp/QSA/Indexer → **V3-Kern ca. 700-800 Zeilen** (Obergrenze, Hunk-Ebene).

### Mechanik [Code]

- Metadaten statt Maske (`fork/src/llama-graph.cpp:43-85`): drei I32-Inputs
  `cell_pos[n_kv]` (Position je KV-Zelle, `INT32_MIN` = leer/fremde Sequenz), `tok_lo[n_tokens]`, `tok_hi[n_tokens]` (sichtbares Positionsintervall je Query). Sichtbar ⇔ `tok_lo[t] <= cell_pos[c] <= tok_hi[t]` (Semantik `fork/ggml/include/ggml.h:2332-2337`). Bei uns: 512 KiB + 2·32 KiB statt 2 GiB.
- Die dichte Maske wird weiter angelegt, aber FA bekommt `kq_mask = nullptr` (`fork/src/llama-graph.cpp:3344-3369`); ohne Konsumenten bleibt sie unalloziert. Upstream 45b6604b2 guardet den Fill schon (`set_input`: `self_attn_mask.raw->buffer`, `llama-graph.cpp:637-640`) — das erleichtert einen Port.
- ggml: `ggml_flash_attn_ext_add_kq_derived()` hängt die Tensoren an `src[6..8]` (`fork/ggml/src/ggml.c:5590-5617`); `GGML_MAX_SRC` ist schon 10, auch upstream.
- Füllen auf dem Host: `llama_kv_cache::set_input_kq_derived` (`fork/src/llama-kv-cache.cpp:3768-3885`), Kausal-Grenze `hi = p1`, SWA standard/chunked über `lo`, non-causal → `hi = INT32_MAX`.
- CUDA/HIP-Kernel (gemeinsamer `ggml-cuda`-Code, keine `#ifdef`-Sperre um den Derived-Zweig):
  - MMA: `fork/ggml/src/ggml-cuda/fattn-mma-f16.cuh:849-900`, `flash_attn_ext_f16_load_mask` erzeugt 0/-inf direkt in `tile_mask` (Laufzeit-Zweig vor dem `cp_async`-Pfad).
  - Tile: `fattn-tile.cuh:1009-1040`. Vec: kein Derived-Zweig, für n_tokens<=8 auch nicht nötig (`fattn.cu:1011-1018`).
  - Kernelwahl sieht „hat Maske“ über `ggml_cuda_flash_attn_ext_has_mask()` = `src[3] || src[5] || src[6]` (`fattn-common.cuh:217-236`, `fattn.cu:684`), damit GQA-Varianten bleiben.
- Host-Seite nicht auf HIP beschränkt: Der Probe-Resolve (`fork/src/llama-context.cpp:1445-1466`) fragt nur `supports_op`.

### Abdeckung / Fallback auf dichte Maske [Code, `fork/src/llama-kv-cache.cpp:3693-3766`, `llama-graph.cpp:63-64`]

| Fall | Verhalten |
|---|---|
| ALiBi | dichte Maske (`use_alibi`) |
| mehrere KV-Streams (`n_seq_max>1` ohne kv_unified) | dichte Maske, Derived global aus |
| `n_tokens <= 8` (Decode, Spec-Verify) | dichte Maske (klein) |
| mehrere Sequenzen, ein Stream | Derived mit Offset `k·2^23` je Sequenz; dicht, wenn ≤32 Token, Token in >1 Sequenz, geteilte Zelle (`seq_cp`) oder Position ≥ 2^23 |
| SWA (standard/chunked) | abgedeckt über `tok_lo`; `can_reuse` bei iSWA aber immer abgelehnt (Graph-Neubau) |
| verschobene Zellen / Prefix-Reuse / Lücken durch leere Zellen | abgedeckt: `cell_pos` = echte `pos_get(j)`, leere/fremde Zellen `INT32_MIN` |
| beliebige -inf-Löcher, die kein Intervall sind | nicht darstellbar; im Standard-Kausalpfad kommen sie nicht vor |
| `n_kv % 256 != 0` | dichte Maske (kommt laut Kommentar nicht vor, `get_n_kv` paddet) |
| **M-RoPE mit echten 2-D-Positionen (Bilder)** | dichte Maske, **und zwar für jeden Batch, solange irgendeine Zelle der Sequenz `ext.x != ext.y` hat**, d. h. auch Text-Prefill nach einem Bild im Kontext (`llama-kv-cache.cpp:3733-3763`) |

Reserve: `kq_mask_packed_reachable()` (`fork/src/llama-context.cpp:1647-1680`) reserviert die dichte Maske, wenn der Aufrufer 2-D-Positionen angekündigt hat (`params.pos_2d`, mmproj) oder ALiBi/Multi-Seq greift; sonst das kleine Derived-Layout, mit Latch + einmaligem Re-Reserve beim ersten 2-D-Batch.
**Für uns relevant:** Unser coding-Profil läuft mit `--mmproj` (`INFERENZSERVER.md:73`). Mit Bildern im Kontext braucht auch der Fork die volle Maske; das Reserve muss sie dann vorhalten oder beim ersten Bild nachwachsen (bei 8 GB → OOM-Risiko). Bild-Support zu behalten und gleichzeitig ub 8192 zu fahren, löst V3 also nur für reine Text-Kontexte.

### Verlust des kausalen Tile-Skips [Code]

- Fork `fattn-common.cuh` ~Z. 2141: `scan_mask` verlangt weiter `mask` ≠ nullptr → bei Derived **kein `KV_max`-Skip**.
- Der Ersatz `flash_attn_kq_derived_blocks` (Bitmap über voll maskierte 256er-KV-Gruppen, batchweit, also nicht pro Q-Tile kausal) ist an `GGML_CUDA_CC_IS_AMD(cc)` gebunden (Z. 2181-2187) → auf NVIDIA gar nichts.
- Ein 1:1-Port auf Ampere rechnet also das obere Dreieck innerhalb jedes ubatch voll mit. Eigene Rechnung: bei 128k-Prompt mit ub 8192 sind das ca. +6 % Attention-FLOPs über den ganzen Prefill, bei Tiefe 0 das Doppelte. Für einen Port müsste man `KV_max` aus `cell_pos`/`tok_hi` ableiten (kleiner Kernel, exakt).

### Benchmarks [Text / fremd gemessen, alles AMD]

- README.md:574-605: 9B dense, `-c 98304`, ub 512: Compute-Buffer **184,0 → 88,4 MiB Device, 112,0 → 16,4 MiB Host**. PP512 an/aus je nach Arch/Tiefe −3,4 % … +3,5 %.
- Hardware laut `archive/work/kq-mask-derived-ab/README.md`: gfx1201 (3× R9700), gfx1151 (Strix Halo), gfx1100 (7900 XTX).
- Unser Modell (Qwen3.6-35B-A3B Q8_0, gfx1151, `data/csv-gfx1151-35B-A3B.txt`), pp512 derived/packed: d0 1835/1843, 32k 1093/1090, 64k 774/781, 98k 598/603 t/s → ±1 %.
- Alles mit ub 512, wo der `KV_max`-Scan (`ne[1] >= 1024`) auch im Referenzarm nicht läuft. Der Verlust aus dem vorigen Abschnitt steckt in diesen Zahlen also nicht drin.
- „Bit-identisch an den Logits“ (`gate-prefill-logits.sh --ab LLAMA_KQ_MASK_DERIVED=0`) ist [Text] (DIAGNOSTIC.md:398, GREEDY-PURITY.md).
- **NVIDIA: [nicht gefunden].** Keine NVIDIA-Messung oder -Validierung zum Derived-Pfad im Repo. AGENTS.md:196 sagt ausdrücklich: „Not validated on NVIDIA is an acceptable, documented state“.

### Portierbarkeit auf unseren Pfad (CUDA, sm_86, D=256, GQA 8, MMA_F16)

- **Machbar, exakt (keine Approximation): Sicherheit hoch.** Prädikat und Füllung sind trivial und modellunabhängig; der MMA-Kernel-Eingriff ist ein Laufzeit-Zweig in `load_mask`, den unser sm_86 genauso nimmt.
- Aufwand, grob mittel: ca. 600-900 Zeilen, 2-4 Tage inkl. Test.
  - Host: graph + kv-cache + cparams/probe, ~400.
  - ggml: op-API + CPU-Referenz in `ops.cpp` für test-backend-ops, ~100.
  - CUDA: `load_mask`-Zweig, Durchreichen von 3 Pointern durch MMA-Launcher und Template-Signaturen, `has_mask`-Prädikat für GQA-Wahl, ~150.
  - Neu und nicht im Fork: derived `KV_max` für NVIDIA, ~50-100.
- Hindernis: Upstream hat `fattn-mma-f16.cuh`/`fattn-common.cuh`/`fattn.cu` seit 45b6604b2 um +840/−405 Zeilen umgebaut (`git diff --stat 45b6604b2 origin/master`). Den Fork-Patch nimmt man deshalb als Vorlage und portiert von Hand, statt ihn per `git am` einzuspielen (er setzt ohnehin 18 RDNA-Blöcke voraus).
- Unbekannt, also messen:
  - Ampere-Performance des Derived-Zweigs: er umgeht `cp_async` für die Maske, liest aber `tok_lo`/`tok_hi` je Zeile aus Global.
  - Ob der Fork-Stand auf NVIDIA überhaupt kompiliert, ist nicht geprüft (nicht gebaut).
- Mit mmproj und Bildern im Kontext hilft es nicht (s. o.). Für ub 8192 + 128k + Bilder müsste man zusätzlich die M-RoPE-ext-Klausel ableiten, die es im Fork nicht gibt.

## 2. ggml-org/llama.cpp PR #29510

- **Existiert, MERGED [Code/gh]:** „llama + CUDA: flash_attn_ext_rows (fix unified kv for multi-seq)“, Autor am17an, gemerged 2026-11-02 16:13 UTC, Merge-Commit **`1b6935b22`** (b12059), +1205/−331.
- **In 45b6604b2 enthalten: ja** (`git merge-base --is-ancestor 1b6935b22 45b6604b2` → true).
- Was es tut [Code, `git show 1b6935b22`]:
  - Neue Op `ggml_flash_attn_ext_rows(q,k,v,mask,kv_rows,…)`: `kv_rows` = I32-Liste der Zellen je Q-Slice.
  - Die Maske hat dann `n_kv_rows` statt `n_kv` Spalten (`llama-graph.cpp` 45b6604b2 Z. 97-127). Auf CUDA nur der MMA-Kernel, NVIDIA (`fattn.cu`: `if (kv_rows) return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && mask … ? MMA_F16 : NONE`).
  - Aktiv nur mit `cparams.auto_fkvr = cparams.kv_unified` und ohne `LLAMA_KV_ROWS_DISABLE` (`llama-context.cpp` im PR-Diff).
- **Spart bei uns nichts. Sicherheit hoch:**
  1. Mit `-np 1` ist `kv_unified` aus (common.h:582 Default `false`; `--kv-unified` nur automatisch bei „slots auto“, arg.cpp:1675-1677), also inaktiv.
  2. Selbst wenn aktiv: `get_n_kv_rows()` (kv-cache 45b6604b2 Z. 1390-1415) zählt die Zellen der Sequenz, gepaddet auf 256. Bei einer Sequenz, die den Cache füllt, ist das = n_kv. Die Maske bleibt `n_kv × n_tokens` F16; der PR ist eine Multi-Seq-Maßnahme (kompakte Spalten je Sequenz), keine Maskenentfernung.

## 3. Weitere Upstream-Arbeiten zur Maske (bis 2026-11-25)

Nur das, was ich im PR oder Code gesehen habe:

- **Gemerged, aber modellspezifisch, hilft uns nicht:**
  - #30965 `qwen4exp : build the QSA selection without the kq_mask`
  - #31009 `glm5-next : QSA without kq_mask (+ DSA)` (gemerged 2026-10-27, `d78f40d9d`). PR-Text: „saves 2 GB … at ub 4096 and 4 GB at ub 8192“ [Text/fremd gemessen]. Gilt nur für QSA/DSA-Attention.
- **Geschlossen, nicht gemerged:**
  - #32752 „[Inkling 2] Build mask in chunks“ (geschlossen 2026-11-24).
  - #32883 „inkling2 : query chunking“, Q-Chunking mit Maske je Chunk, `LLAMA_GRAPH_CHUNK` (geschlossen 2026-11-25), plus Ableger #32884-#32887 für qwen4exp, deepseek41, kimi-k31, glm5-next; alle geschlossen.
  - #30815 (qwen4exp QSA chunking, Prototyp, geschlossen).
  - Konzeptionell wäre Q-Chunking eine zweite exakte Variante (Maske nur `n_kv × chunk`). Upstream gibt es das aber nicht, und es war nie für qwen35moe.
- #32740 „ggml: compact bias on the existing FLASH_ATTN_EXT“ (gemerged 2026-11-23, `239242322`): Indexed Bias für Inkling 2, CUDA lehnt ab (`fattn.cu`: `has_indexed_bias → NONE`). Es ist keine Maskenreduktion.
- #31600 (offen): „ordered KQ masks“ beschleunigt nur den Host-Fill, kein Speichergewinn.
- #32535 / #31536 / #32394: Masken-Broadcast über Q-Batch bzw. Heads in WebGPU/SYCL, kein CUDA- und kein Speicherthema.
- Upstream-master `build_attn_inp_kq_mask` (Stand `9c003a97a`): weiterhin dichte `n_kv × n_tokens`-F16-Maske. **Eine generische Derived/Implicit-Causal-Maske für FA: upstream [nicht gefunden].**
