# Fremde Implementierungen zu Prefill-Vorabladen und Layer-Major (gelesen am 2026-11-25)

Gelesen wurde Code bzw. Doku, nichts davon ist auf der Testmaschine gebaut oder gemessen.

## 1. thecodacus `fable5/prefetch-experts` (zu Issue ggml-org/llama.cpp#25859): passt am besten

- Der Branch hat 4 Commits und ist 2904 hinter master. Die Kernänderung umfasst ~170 Zeilen in `ggml/src/ggml-backend.cpp`, dazu ~60 Zeilen Pinning in `llama-mmap`/`llama-model-loader`.
- `GGML_SCHED_PREFETCH_EXPERTS=N`: Für Splits, deren erster Knoten `MUL_MAT_ID` mit Host-Gewichten ist, und ab Prefill-Größe
  (`n_used·n_tokens ≥ 2·n_expert`) wird der **ganze** Expertentensor über eine zweite Backend-Instanz (zweiter CUDA-Stream)
  in einen von N Staging-Slots geladen. Der D2H-Readback der Routing-ids entfällt, damit gibt es im Loop keinen Sync.
  Synchronisiert wird mit Events (ready/free je Slot).
- Die Slots haben die Größe des **größten** Expertentensors, Standard sind 3 (gate/up/down eines Layers). Pro Layer gehen ~0,48 GB über den Bus.
  Laut auswertung.md sind gate+up 214/370 der Bytes, down ist also mit ~0,2 GB der größte Tensor. 3 Slots kosten damit ~0,6 GB,
  etwas mehr als der geschätzte zweite Puffer und knapp unter den ~830 MiB, die neben coding frei sind. Geschätzt, nicht gemessen.
- `GGML_CUDA_REGISTER_HOST=1` macht `cudaHostRegister` auf den gemappten Bereich, also den mmap-Pfad ohne Bounce-Buffer.
  Bei uns erledigt das schon `coding-dio` (gepinnt). Für `coding` (mmap) hieße es ~20 GB zu pinnen, praktisch also dio.
- Gemeldet sind 1143 → 1880 t/s (+64 %) auf einer RTX 3060 mit Gen4 und Qwen3.6-35B-A3B bei ub 2048. Davon kommen ~+21 % vom Pinning,
  das Vorabladen allein bringt also ~+36 %. Unser Modell für dio sagt +43–48 %, das liegt im selben Bereich.
- Für uns gilt der Readback-Befund des Autors: Bei pp2048 werden 256/256 Experten benutzt, also derselbe Befund wie unsere 94–98 %.
- Ein Port als PR #28414 (`--prefetch-experts-slots`, max. 4 Slots) wurde geschlossen (Template, AI-Hinweis). Ein Bot-Review fand
  einen Blocker: Prefetch-Zustand je Scheduler statt je Device, dadurch ist Multi-GPU `-sm layer` kaputt. Für uns mit einer GPU spielt das keine Rolle.
  Der Autor meldet +22–30 % bei 35B-MoE und keinen nennenswerten Gewinn über 3 Slots.

## 2. PR ggml-org/llama.cpp#21067 `--prefetch-weights` (am17an, Draft, offen seit 2026-03-27)

- Allgemeiner gebaut: Lädt alle überschriebenen Gewichte eines Layers vorab (dense und MoE) und braucht `--no-mmap`.
- Reserviert wird je ganzem fused Split mit `n_copies = 1`. Laut mobilinkd im Issue staut die Pipeline damit oft. Er hat mit dem
  slot-basierten Code von thecodacus +17–20 % gemessen, mit #21067 weniger.
- Messungen Dritter sind gemischt: noonghunna (2×3090, 117B MoE) +14 % Prefill, aber TTFT +48 % bei kurzen Prompts. Ursache ist der Verlust
  der selektiven Kopie, behebbar mit einem Batch-Gate. crusaderky (3090, **PCIe 3.0**, ub 2048) misst **150 → 95 t/s, also langsamer**.
  Warum, ist im Thread nicht geklärt.
- Der VRAM für den Doppelpuffer wird für den Worst-Case-Graphen reserviert, und zwar dauerhaft.
- Schlussfolgerung: Für unseren Fall ist das die schlechtere Basis. Die slot-basierte Variante aus 1. ist kleiner und zielt genau auf
  MoE-Experten.

## 3. Anemll `anemll-flash-llama.cpp` `--moe-prefill-layer-major`: **keine Wiederverwendung über ubatches**

- Der Fork ist für Apple Silicon gebaut und streamt Experten von der SSD in Sidecar-Bänke, einen CUDA-Pfad gibt es nicht.
- „Layer-major“ heißt dort, dass innerhalb **eines** Prompt-Slabs die Experten-Referenzen aller Token dedupliziert werden. Jeder
  einzigartige Experte wird einmal geladen und für alle seine Token benutzt. Laut Doku (Abschnitt „Reuse Model“) gibt es ausdrücklich
  **keine** persistente Wiederverwendung über Layer oder über Prompt-Slabs.
- Genau das macht llama.cpp auf CUDA schon je ubatch: Jeder benutzte Expertentensor geht einmal pro ubatch über den Bus und
  bedient alle 4096 Token. Das `--moe-prefill-batch` des Forks entspricht bei uns `-ub`. Für die Testmaschine steckt darin kein neuer Hebel.

## 4. Echte Layer-Major-Wiederverwendung über ubatches (Vorschlag aus der Diskussion)

- Gemeint ist: einen Layer einmal hochladen und alle ubatches des Prompts hindurchschicken. Bei 64k sind das 16 ubatches, der Gewichtsverkehr
  sinkt also ideal auf 1/16. Bei 4k (ein ubatch) bringt es **nichts**.
- Kausal ist das zulässig. Attention von ubatch k in Layer l braucht nur den KV von ubatch 0..k in Layer l, und der DeltaNet-Zustand
  läuft innerhalb des Layers in Token-Reihenfolge weiter. Beides geht in Layer-Major-Reihenfolge auf.
- Eine CUDA-Implementierung in llama.cpp habe ich nicht gefunden. Der Umbau wäre tief: Graph je Layer statt je ubatch, Hidden States
  aller Token zwischenspeichern (FP32 bei 64k: 512 MiB, notfalls im Host-RAM) und das KV-/Recurrent-State-Handling in `llama-context` anpassen.
- **Billiger Teilhebel ohne Code:** ein größeres `-ub`. Bei ub 8192 halbiert sich der Upload je Token für Prompts ≥ 8k. Begrenzt wird das
  durch den Compute-Buffer im VRAM (llama-bench wollte bei 64k schon mit ub 4096 4570 MiB, neben coding sind ~830 MiB frei).
  Das ist nicht gemessen.

## Konsequenz für die nächsten Schritte

1. Den Scheduler-Teil von thecodacus (nur `ggml-backend.cpp`) auf den Stand von `~/src/llama.cpp-prof` portieren und mit `coding-dio` und
   `GGML_SCHED_PREFETCH_EXPERTS=3` gegen die Basis messen (4k/16k/64k, echter Text, Token-Identität bei temp 0).
   Das klärt zugleich, ob die 3 Slots neben coding ins VRAM passen.
2. Danach `-ub 8192` mit und ohne Prefetch, sofern der Compute-Buffer passt.
3. Den mmap-Pfad erst angehen, wenn 1. trägt. Er bräuchte dann das Pinning aus dem Branch, und das ist faktisch dio.
