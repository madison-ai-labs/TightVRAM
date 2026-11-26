# ubatch-Grenze und Prefetch-Port (Lauf 3 und 4, 2026-11-25)

Bedingungen wie in Lauf 2: llama-server mit den exakten `LLAMA_ARGS` des Profils, geändert sind nur `-b`/`-ub`/`-c`.
Echter Text (`text.txt`), alle Konfigurationen bekommen dieselben Textabschnitte, `cache_prompt: false`.
Lauf 3 lief über das Dienst-Binary (ungepatcht), Lauf 4 über den pf-Build.

## Prefetch-Port

- `~/src/llama.cpp-pf`: git worktree von 45b6604b2 (= Dienst-Build b12156), Patch `patch-prefetch-port.diff` (+187 Zeilen,
  nur `ggml-backend.cpp`). Erzeugt mit `port-prefetch.py`.
- Logik von thecodacus `fable5/prefetch-experts` (5e7f627). Upstream hat die Input-Kopie seither nach
  `ggml_backend_sched_copy_input` ausgelagert, samt `callback_copy` für die selektive Expertenkopie aus llama-context.
  Der Prefetch-Pfad sitzt deshalb in der Host-Gewichte-Schleife von `compute_splits` und umgeht diesen Callback, also auch den ids-Readback.
- An/aus über `GGML_SCHED_PREFETCH_EXPERTS=3` (3 Slots). Ohne die Variable ist der Build identisch zum Dienst:
  ref 1643/1629/1418 t/s gegen Dienst-Binary 1637/1630/1422 (Lauf 3).
- Der mmap-Pfad bekommt kein Pinning (der `GGML_CUDA_REGISTER_HOST`-Teil ist nicht portiert).

## Ergebnisse (t/s Prefill, Mittel; kalt = erster Request nach dem Laden)

| Profil | ub | ctx | Prefetch | kalt 4k | 4k | 16k | 64k | 120k | tg 16k | VRAM Peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| coding-dio | 4096 | 131072 | nein | 1542 | 1643 | 1629 | 1418 | 1237 | 31,0 | 7163 |
| coding-dio | 4096 | 131072 | **ja** | 2259 | **2384** | **2438** | **2051** | **1696** | 30,5 | 7793 |
| coding-dio | 8192 | 65536 | nein | 1539 | 1645 | 2285 | 1917 | – | 31,0 | 6657 |
| coding-dio | 8192 | 65536 | **ja** | 2213 | 2384 | **3590** | **2768** | – | 32,5 | 7285 |
| coding (mmap) | 4096 | 131072 | nein | 1204 | 1408 | 1402 | 1241 | – | 32,5 | 7123 |
| coding (mmap) | 4096 | 131072 | **ja** | 1849 | 1934 | 1983 | 1740 | – | 31,4 | 7753 |

64k heißt hier 65024 Token, damit der Prompt auch in `-c 65536` passt. Rohdaten: `roh4/ergebnis.jsonl`, Auswertung: `auswertung-lauf4.txt`.

- **Prefetch:** dio +45 / +50 / +45 / +37 % (4k/16k/64k/120k), mmap +37 / +41 / +40 %. Das liegt genau im Modell aus
  auswertung.md (dio +43–48 %). Der 120k-Wert ist niedriger, weil dort die Attention einen größeren Anteil hat.
- **Offene Frage zu mmap geklärt:** Der Upload aus nicht gepinntem Speicher überlappt mit den Kernels.
  Ob `cudaMemcpyAsync` aus pageable Speicher dabei den Host-Thread blockiert (laut CUDA-Doku möglich), ist nicht gemessen. Gemessen ist nur das Ergebnis: Der zweite Stream läuft parallel zum Rechnen.
  Der Abstand dio → mmap bleibt mit ~20 % bestehen (Bandbreite 9,9 gegen 12,3 GB/s).
- **ubatch 8192** (ohne Code) bei 16k: 1629 → 2285 (+40 %), bei 64k: 1418 → 1917 (+35 %). Bei 4k bringt es nichts (ein ubatch).
- **Kombination** dio + ub 8192 + Prefetch: 16k **3590 t/s (2,2× heute)**, 64k **2768 (1,95×)**.
- **VRAM:** Prefetch kostet +630 MiB (3 Slots à größter Expertentensor). Bei dio ub 4096 und 128k bleiben 7793 von 8192 belegt,
  ~0,4 GB frei. Das ist knapp, lief aber stabil, auch bei 120k.
- **Decode** nach dem Prefill (64 Token, temp 0, 16k Kontext): Der Text ist in allen drei Paaren ref/pf identisch, die Top-10-Logprobs
  sind **bitgleich** (max |Δ| = 0). tg 30,5–32,5 t/s ohne Richtung. Prefetch greift erst ab `n_used·n_tokens ≥ 2·n_expert`
  und ist im Decode also aus.
- **Kalt gegen warm:** Der erste Request nach dem Laden ist 5–14 % langsamer (Graph-Reserve, erste Kernel). Prefetch hilft dort genauso.
- **RAM:** dio belegt ~22,4 GB, unverändert durch Prefetch (die Slots liegen im VRAM).

## ubatch-Grenze (Compute-Buffer)

| ub | ctx | lädt | Compute-Buffer, der scheitert |
|---|---|---|---|
| 4096 | 131072 | ja | – |
| 6144 | 131072 | **nein** | 2978 MiB |
| 8192 | 131072 | **nein** | 3842 MiB |
| 8192 | 65536 | ja | – |
| 12288 | 65536 | lädt, **stürzt beim ersten Request ab** (`CUDA error: the resource allocation failed`) | – |

- Bei 128k-Kontext ist **ub 4096 die Grenze**. Bei 64k-Kontext geht ub 8192, 12288 nicht mehr.
- Steigung: +864 MiB Compute-Buffer je +2048 ubatch bei ctx 131072. Ein Teil davon ist sicher die KQ-Maske. Im Code
  (`llama-graph.cpp:30-47`, **korrigiert in Lauf 5**: 1098 ist der DSV4-Pfad) hat sie die Form `n_kv × n_tokens` in F16 (bei Flash-Attention), bei 128k also 512 MiB je 2048 Token, ~60 % der Steigung.
  Rechnerisch belegt sie bei ub 4096 und 128k 1 GiB, bei ub 8192 2 GiB. Dass `n_kv` bei der Reserve den ganzen Kontext nimmt,
  war hier aus der Skalierung geschlossen. Lauf 5 hat es im Allocator bestätigt, siehe `deploy-und-compute-buffer.md`. Die Aufteilung des Rests (Aktivierungen, Experten-Staging)
  ist offen. Das ist Schritt 3 (Compute-Buffer-Analyse).

## Konsequenz

1. Prefetch ist die belastbare Verbesserung: +37–50 %, bitgleiche Ausgabe, 128k bleibt nutzbar. Gekostet hat das 187 Zeilen.
2. ub 8192 bringt zusätzlich bis +47 % (16k), aber nur mit 64k-Kontext. Ob es bei 128k geht, hängt an der KQ-Maske bzw. dem
   Compute-Buffer.
3. Bevor das in den Dienst geht: Profil mit pf-Build (`/opt`-Deployment, `GGML_SCHED_PREFETCH_EXPERTS=3` im Unit-Env) und eine
   VRAM-Reserve von 0,4 GB. Das ist eine Betriebsentscheidung.
