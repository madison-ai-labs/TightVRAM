# Prefill des Dienstmodells: Anteil Experten-Upload, lohnt Überlappen? (2026-11-25)

Modell und Argumente wie dem Produktionsprofil `coding` bzw. `coding-dio` (Argumente siehe README)
(Qwen3.6-35B-A3B UD-Q4_K_XL, `-ngl 99 -ncmoe 40 -fa on -t 10 -b 4096 -ub 4096 -c 131072 -np 1 --fit off --cache-ram 4096`,
coding mit `--mmproj … --no-mmproj-offload` und Load-Mode auto = mmap, dio mit `--load-mode dio`).
Build 45b6604b2 (= `/opt/llama.cpp/current`, b12156). Am Scheduler ist nichts umgebaut, es ist nur gemessen.

## Ergebnis

1. **Echter Prefill** (llama-server, echter Text, Dienst-Binary): coding (mmap) 1409 / 1403 / 1249 t/s,
   coding-dio 1650 / 1618 / 1420 t/s bei 4k / 16k / 64k. dio ist um 17 / 15 / 14 % schneller.
2. **Zeitanteile:** Der Prefill ist Upload-gebunden. Der H2D-Upload der Experten macht **65 % (mmap) bzw. 59 % (dio)**
   der Prefill-Zeit aus, bei 64k 59 / 54 %. Das reine Expertenrechnen (MUL_MAT_ID gate/up/down) kostet nur 10 bis 13 %,
   Attention/DeltaNet/Shared Expert/Router 22 bis 32 %. Je 4096 Token werden ~18,7 GB hochgeladen, mit 9,9 GB/s (mmap, pageable)
   bzw. 12,3 GB/s (dio, gepinnt). 12,3 GB/s ist praktisch das Limit von PCIe 3.0 x16. Der Link schafft maximal Gen3
   (`pcie.link.gen.max 3`, Karte könnte Gen4), schneller übertragen geht also nicht.
3. **Expertennutzung bei ub 4096, echter Text:** 98 % der 256 Experten je Layer und ubatch im Mittel (64k-Anfrage,
   min 90 %). Bei der 16k-Anfrage waren es 94 % (min 88 %). Mit Zufallstoken (llama-bench) sind es nur 85 %, das ist für diese
   Frage nicht repräsentativ. Alle Experten vorab zu laden kostet also nur 2 bis 6 % mehr Bytes und braucht das Routing
   nicht. **Vorabladen ohne Router-Abhängigkeit geht.**
4. **Was Überlappen bringt** (Modell aus den gemessenen Anteilen, Tabelle unten):
   - Überlappen innerhalb eines Layers mit dem heutigen routing-abhängigen Upload (up/down hochladen, während gate/up rechnen)
     bringt nur **+6 bis 8 %**. Das Expertenrechnen ist zu klein, um viel zu verstecken.
   - Wer alle Experten von Layer l+1 auf einem zweiten Stream vorab lädt, während Layer l rechnet, gewinnt
     **dio +43 / +46 / +48 %**. Die Basis ist der Profil-Lauf (1577 → 2257 t/s bei 4k, 1420 → 2104 bei 64k, Tabelle unten).
     Auf die Server-Messung aus Ergebnis 1 übertragen heißt das 1650 → ~2360 t/s bei 4k und 1420 → ~2100 bei 64k.
     (Korrigiert 2026-11-25: Vorher stand hier „1650 → 2260“. Das mischte die Server-Basis mit dem Endwert aus dem Profil-Lauf.)
     Das ist die Schranke für gepinnten Speicher. Auch dann bleibt der Prefill Upload-gebunden: In den 30 DeltaNet-Layern ist der Upload
     (~39 ms) länger als das Rechnen (~21 ms). Nur in den 10 Attention-Layern verschwindet er bei langem Kontext ganz.
   - Für mmap rechnet das Modell +36 / +36 / +44 %, **aber das gilt nur, wenn ein pageable-Upload neben laufenden
     Kernels wirklich parallel läuft. Gemessen ist das nicht** (Synthetik und dieser Lauf messen pageable nur seriell).
     Mit `cudaMemcpyAsync` aus pageable Speicher blockiert der Host. Realistisch braucht der mmap-Pfad einen eigenen gepinnten
     Staging-Puffer. Damit wäre der mmap-Vorteil (kein 20-GiB-Pin) gegen eine zusätzliche CPU-Kopie getauscht.
   - Eine Orakel-Variante, die nur die benutzten Experten lädt, aber vorab, gewinnt kaum mehr als „alle laden“ (+2 bis 8 Punkte).
     Weil 94 bis 98 % ohnehin benutzt werden, lohnt Routing-Vorhersage nicht.
   - **Kosten in VRAM:** ein zweiter Puffer für einen Layer mit allen Experten, ~0,48 GB. coding belegt 7009 von 7839 MiB,
     frei sind ~830 MiB. Auf dem Papier passt das, ausprobiert ist es nicht.

Fazit: Das Überlappen lohnt sich, aber nur als Vorabladen aller Experten der nächsten Schicht auf einem zweiten Stream.
Das Umsortieren des heutigen Uploads innerhalb des Layers bringt <10 %. Der Gewinn ist für dio belastbar (gepinnt, +43 bis 48 %).
Für mmap braucht es zuerst den Nachweis, dass pageable-Upload überhaupt parallel zu den Kernels läuft, oder einen Staging-Puffer.

## Messaufbau

- **Messbuild** `~/src/llama.cpp-prof` auf der Testmaschine: git worktree von 45b6604b2, gleiche CMake-Optionen wie der Dienst-Build
  (CUDA, arch 86, FA, native, Release), Patch `patch-prof.diff`. Er liegt neben dem Dienst und nicht in `/opt`.
  Mit `LLAMA_PREFILL_PROF=1` misst er:
  - in `ggml_backend_sched_compute_splits` je Split die Aktivierungs-Kopien, den Upload der Host-Gewichte (Experten) und das Rechnen,
    jeweils mit `ggml_backend_synchronize` abgegrenzt. In Splits mit Host-Gewichten wird der erste Knoten (MUL_MAT_ID) als
    eigener Graph-View gerechnet, damit Expertenrechnen und der Rest des Splits getrennt sind. Der down-Split enthält
    die Attention/DeltaNet der nächsten Schicht (99 bzw. 72 Knoten, siehe `roh/C-auto-4096.err`, `LLAMA_PREFILL_PROF=2`).
    Ausgabe: eine `PFX U`-Zeile je Graph-Compute.
  - in `llama_context::sched_copy_experts` je Expertentensor die Zahl der benutzten Experten aus den ids, die Bytes
    und die Zeit für ids holen und H2D (`PFX E`).
  - **Overhead:** llama-bench 16k mit Timern 1503 t/s, ohne 1511 t/s (−0,5 %). Server-Messbuild gegen Dienst-Binary
    zwischen 0,96 und 1,01, ohne Richtung.
- **Lauf 1** (`lauf.sh`): llama-bench mit Zufallstoken, Dienst-Binary und Messbuild.
  Bei 64k bricht llama-bench in beiden Load-Modes mit OOM ab: Der Compute-Buffer will 4570 MiB, der Server mit
  128k braucht das nicht. Die Ursache ist nicht untersucht, sie liegt in der Kontext-Reserve von llama-bench. Ebenso scheitert
  `llama-perplexity -c 65536`. Deshalb gibt es Lauf 2.
- **Lauf 2** (`lauf2.py`): `llama-server` (Dienst-Binary = ungepatchter Build, dann
  Messbuild) mit den exakten Serving-Argumenten des Profils. Text: llama.cpp-Quelltext plus Markdown
  (`text.txt`, 3 MB, 1,04 M Token), als Token-Array gesendet, jede Anfrage mit eigenem Textabschnitt, `cache_prompt: false`,
  `n_predict: 1`, die Zeit aus `timings.prompt_ms`. Je Konfiguration ein Warmup von 4096 Token, verworfen, dann 3 Wiederholungen
  (Dienst) bzw. 1 (Messbuild) je Länge. Der Server rechnet die letzten 4 Token eines Prompts als eigenen ubatch
  (Context-Checkpoint), ~100 ms mit Experten auf der CPU. Der ist in den Summen enthalten.

## 1. Echter Prefill

### llama-server, echter Text (Lauf 2)

| Profil | Prompt | Dienst-Binary t/s (Mittel, Einzelwerte) | Messbuild t/s |
|---|---|---|---|
| coding | 4096 | 1409 (1398, 1437, 1390) | 1382 |
| coding | 16384 | 1403 (1415, 1400, 1395) | 1421 |
| coding | 65536 | 1249 (1251, 1248, 1248) | 1257 |
| coding-dio | 4096 | 1650 (1618, 1674, 1656) | 1577 |
| coding-dio | 16384 | 1618 (1639, 1611, 1605) | 1639 |
| coding-dio | 65536 | 1420 (1421, 1419, 1421) | 1420 |

dio/mmap: 1,171 (4k), 1,153 (16k), 1,137 (64k).

### llama-bench, Zufallstoken (Lauf 1, Dienst-Binary, -r 3)

| Prompt | auto (mmap) | dio | dio/auto |
|---|---|---|---|
| 4096 | 1563 ± 5 | 1838 ± 9 | 1,176 |
| 16384 | 1510 ± 6 | 1770 ± 4 | 1,172 |
| 65536 | OOM | OOM | |

llama-bench liegt 10 % über dem Server. Mit Zufallstoken laufen nur 85 % der Experten über PCIe statt 98 %.

## 2. Zeitanteile (Lauf 2, Messbuild, je 4096 Token)

| Profil | Prompt | je 4k ms | up | exp | attn+ | in | außer | up % | exp % | attn+ % | GB H2D | GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coding | 4096 | 2963 | 1912 | 322 | 649 | 13 | 67 | 65 % | 11 % | 22 % | 18,72 | 9,9 |
| coding | 16384 | 2882 | 1870 | 314 | 627 | 16 | 56 | 65 % | 11 % | 22 % | 18,02 | 9,8 |
| coding | 65536 | 3258 | 1923 | 315 | 931 | 32 | 58 | 59 % | 10 % | 29 % | 18,82 | 9,9 |
| coding-dio | 4096 | 2597 | 1545 | 320 | 653 | 13 | 65 | 59 % | 12 % | 25 % | 18,72 | 12,3 |
| coding-dio | 16384 | 2499 | 1485 | 314 | 628 | 16 | 56 | 59 % | 13 % | 25 % | 18,02 | 12,3 |
| coding-dio | 65536 | 2885 | 1547 | 315 | 933 | 32 | 58 | 54 % | 11 % | 32 % | 18,82 | 12,3 |

up = ids holen (je Layer ~0,5 ms, Sync) + H2D der benutzten Experten. exp = MUL_MAT_ID gate/up/down + SWIGLU.
attn+ = übriges Rechnen der Layer (Attention bzw. Gated DeltaNet, Shared Expert, Router, Norms) und der 4-Token-Rest-ubatch.
in = Aktivierungs-Kopien. außer = prompt_ms minus Graph-Zeit (Graph bauen, Inputs, Server).

Verlauf über die 16 ubatches der 64k-Anfrage, dio (ubatch / up / exp / attn+ ms): 2394/1545/317/521 … 3271/1558/314/1347.
Upload und Expertenrechnen bleiben konstant, nur attn+ wächst mit der KV-Länge um ~55 ms je ubatch. Der Zuwachs liegt in den 10
Full-Attention-Layern (3, 7, …, 39). Rohwerte für alle ubatches in `auswertung-lauf2.txt`.

Der Upload eines Layers mit allen Experten kostet gepinnt real ~39 ms. Die Synthetik sagte 36,5 ms
(`tensor-transport-20261125`, Q4_K ub 4096, 43,8 − 7,3), das passt. Expertenrechnen je Layer ~7,9 ms, Synthetik 7,3 ms.

## 3. Expertennutzung je Layer und ubatch (ub 4096, 256 Experten)

| Quelle | ubatches | benutzt Mittel | Median | min | max | Layer-ubatches mit allen 256 |
|---|---|---|---|---|---|---|
| echter Text, 64k-Anfrage | 16 | 98,3 % | 98,4 % | 89,8 % | 100 % | 14,3 % |
| echter Text, 16k-Anfrage | 4 | 94,1 % | 94,5 % | 87,9 % | 99,2 % | 0 % |
| echter Text, 4k-Anfrage | 1 | 97,8 % | 97,7 % | 94,9 % | 100 % | 7,7 % |
| Zufallstoken (llama-bench), 16k | 4 | 84,8 % | | 58,6 % | 100 % | 1,9 % |

Die 16k-Anfrage stammt aus einem anderen Textabschnitt als die 64k-Anfrage. Die schwächsten Layer dort (32 bis 38) haben im Mittel
232 von 256 benutzt. gate, up und down eines Layers teilen sich die ids, die Zahlen gelten für alle drei.

## 4. Abschätzung Überlappung

Modell je Layer und ubatch. Upload je Layer kommt aus den `PFX E`-Zeilen (gemessen, ids inklusive). Rechnen je Layer: exp/40 + in/40 + b,
in den 10 Full-Attention-Layern zusätzlich g. b = attn+ des ersten ubatch / 40, g = (attn+ − 40·b)/10. Rest-ubatch und
„außer“ bleiben unverändert.

- **A** routing-abhängig, innerhalb des Layers: Upload von up/down läuft, während gate/up rechnen. Spart je Layer das Rechnen von gate
  und up (Anteil an exp nach Bytes 214/370 = 58 %).
- **B** alle 256 Experten von Layer l+1 auf einem 2. Stream vorab laden, während Layer l rechnet:
  T = U_all(0) + Σ max(R(l), U_all(l+1)), U_all = gemessener Upload × (alle Bytes / benutzte Bytes).
- **C** wie B, aber nur die benutzten Experten (Orakel, nicht baubar, obere Schranke).

| Profil | Prompt | T gemessen ms | A | B (alle) | C (Orakel) | nur Rechnen | Gewinn A | Gewinn B | Gewinn C | t/s jetzt → B |
|---|---|---|---|---|---|---|---|---|---|---|
| coding | 4096 | 2963 | 2777 | 2186 | 2142 | 1051 | +7 % | +36 %\* | +38 %\* | 1382 → 1874\* |
| coding | 16384 | 11528 | 10799 | 8484 | 8018 | 4049 | +7 % | +36 %\* | +44 %\* | 1421 → 1931\* |
| coding | 65536 | 52135 | 49216 | 36301 | 35837 | 21370 | +6 % | +44 %\* | +45 %\* | 1257 → 1805\* |
| coding-dio | 4096 | 2597 | 2411 | 1815 | 1780 | 1052 | +8 % | +43 % | +46 % | 1577 → 2257 |
| coding-dio | 16384 | 9994 | 9267 | 6852 | 6485 | 4053 | +8 % | +46 % | +54 % | 1639 → 2391 |
| coding-dio | 65536 | 46155 | 43234 | 31143 | 30788 | 21397 | +7 % | +48 % | +50 % | 1420 → 2104 |

\* nur unter der Annahme, dass ein pageable-Upload parallel zu den Kernels läuft. Ungemessen, siehe Ergebnis 4.

Annahmen, die das Modell nicht prüft: Der Upload behält auf einem zweiten Stream neben den Kernels seine 12,3 GB/s
(der VRAM-Bandbreitenbedarf von 12 GB/s gegen 448 GB/s spricht dafür), und die zwei Streams kosten keine nennenswerte Sync-Zeit.
Das Rechnen je Layer ist zwischen DeltaNet- und Attention-Layern bei kurzer KV gleich verteilt (C-Lauf: down-Split 14 bis 16 ms
in beiden Typen).

## Offen

- pageable-H2D parallel zu Kernels auf einem zweiten Stream: synthetisch messen (der Benchmark `tensor-transport` hat das
  Werkzeug), bevor für mmap etwas gebaut wird.
- Passt der zweite Expertenpuffer (~0,48 GB) neben coding mit 128k und mmproj in die ~830 MiB?
- Der OOM von llama-bench bei 64k (Compute-Buffer 4570 MiB) ist nicht untersucht.

## Dateien

- `lauf.sh`, `lauf.log`, `roh/`: Lauf 1 (llama-bench, Zufallstoken; `A-*` Dienst-Binary, `B-*` Messbuild mit Timern,
  `B0-*` Messbuild ohne Timer, `C-*` Split-Struktur, `D-*` perplexity OOM)
- `lauf2.py`, `lauf2.log`, `roh2/`: Lauf 2 (llama-server, echter Text; `ergebnis.jsonl`, Server-Logs mit `PFX`-Zeilen,
  in `ergebnis.jsonl` per Byte-Offset `log_a`/`log_b` je Anfrage abgegrenzt)
- `auswert.py` → `auswertung-lauf1.txt`, `auswert2.py` → `auswertung-lauf2.txt` (`python3 -I auswert2.py roh2/`)
- `patch-prof.diff`: der Messbuild-Patch gegen 45b6604b2
- `text.txt` liegt nur auf der Kiste (nicht im Repo), die Erzeugung steht unten

Text erzeugt mit:
`cd ~/src/llama.cpp && cat src/*.cpp src/*.h docs/*.md docs/*/*.md tools/server/*.cpp | head -c 3000000 > text.txt`
