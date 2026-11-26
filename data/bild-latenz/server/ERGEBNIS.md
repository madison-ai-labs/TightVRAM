# Bild-Latenz auf der Testmaschine: wo die Zeit hingeht und was sie verkürzt

Stand 2026-11-11, Messung 11:30 bis 13:11. RTX 3070 8 GB, i9-10850K, 32 GB RAM. llama.cpp b12156, Commit `45b6604b2`, Binary `/opt/llama.cpp/b12156-45b6604b2/llama-server`. Modell `Qwen3.6-35B-A3B-UD-Q4_K_XL`, Profil `coding` (128k, `-ncmoe 40`, `-ub 4096`, `-t 10`, Projektor F16 auf der CPU).

Kennzeichnung: **gemessen** (mit Rohdatei unter `raw/`), **gelesen** (Quelltext, Datei:Zeile im Commit 45b6604b2), **vermutet**, **Urteil** (meine Bewertung einer Antwort).

Ablage: Skripte `scripts/`, Rohdaten `raw/*.jsonl` (je Anfrage eine Zeile mit timings, usage, Antworttext und den Journalzeilen des Messfensters), Tabellen `auswertung/*.md` (erzeugt mit `python3 -I scripts/auswerten.py .`), Ausfälle `ausfall.md`.

## Kurzfassung

1. **Der Encoder ist der Engpass, nicht das LLM.** 1280 px (1000 Bild-Token): Encoder 9,5 s, LLM-Prefill der Bild-Token 2,0 s, Rest 0,25 s. 2560 px (4000 Token): Encoder 81,4 s, LLM nur 2,8 s. **gemessen**, `raw/p1_split.jsonl`, `raw/lv4.jsonl`.
2. **Die Encoder-Zeit wächst überlinear** (etwa Token^1,4; 4fache Token kosten 8,5fache Zeit). Der LLM-Anteil wächst kaum (1,0 s bei 160 Token, 2,8 s bei 4000). **gemessen**.
3. **Projektor auf der GPU (Q8_0, Kontext 64k):** Encoder 1280 px 0,52 s statt 9,5 s, gesamt 2,9 s statt 11,9 s; 2560 px 8,3 s statt 84,5 s. Startete sauber, VRAM-Spitze 6573 MiB. Preis: 64k statt 128k Kontext. **gemessen**, `raw/p5_q8.jsonl`.
4. **Vorwärmen trifft den Cache nur, wenn das Bild eine eigene user-Nachricht ist.** Bild und Frage in derselben Nachricht: kein Treffer, cache_n 0, volle 11,8 s, egal in welcher Reihenfolge. Bild als eigene Nachricht, danach Frage als neue Nachricht: cache_n 1014, prompt_ms 0,37 s. **gemessen**, `raw/p3_prewarm.jsonl`; Ursache **gelesen** (kein Checkpoint nach einem Bild-Chunk).
5. **Kleiner Text kippt zwischen 1280 und 1024 px.** Leiter 2560: 46 von 48 Token richtig, 1280: 31, 1024: 16, 768: 7, 512: 1. Code-Schlüsselstellen ab 768 px fehlerhaft, 512 px erfindet Code. **gemessen**, `raw/p2_quality.jsonl`.
6. **OCR-Text statt Bild:** Zeit bis zur ersten Antwort 1,8 s statt 11,9 s, aber 11 statt 17 von 20 Inhaltsfragen richtig (Diagramm 0 von 4). OCR-Text plus 512-px-Bild: 3,9 s, 14 von 20. **gemessen** und **Urteil**, `raw/p4_ocrfragen.jsonl`.

## Methode

- Alle Anfragen von der Box an `http://127.0.0.1:8080/v1/chat/completions`, nicht streamend (außer wo genannt), `temperature 0`, `enable_thinking: false`, `cache_prompt` Default. Latenzfrage: „Antworte mit einem Wort: Was zeigt das Bild?“, `max_tokens 8`. Bibliothek `scripts/blib.py`.
- **Kalt heißt kalt:** Jede erste Messung bekommt eindeutige Datei-Bytes (JPEG: Zufallsbytes hinter EOI, PNG: zusätzlicher tEXt-Chunk). Pixelgleichheit mit `sips` geprüft. cache_n war bei jeder kalten Messung 0. **gemessen**.
- **Fremdanfragen:** Vor jeder Messung Slot über `/slots` frei, danach Journal des Messfensters geprüft (mehr als ein `processing task` = verwerfen). Es wurde keine Messung verworfen. **gemessen**.
- **Aufteilung Encoder gegen LLM-Prefill:** Bei `-lv 3` (Profil `coding`) schreibt der Server keine Bild-Zeitmarken. Die mtmd-Meldungen sind INFO, der Server leitet sie über `common_log_default_callback`, und dort wird INFO auf Stufe TRACE (4) abgebildet (**gelesen**, `common/log.cpp:551`, Ausgabe in `tools/mtmd/mtmd-helper.cpp:175,193`, `tools/server/server-context.cpp:897`). Deshalb zwei Wege:
  - Ohne Neustart: Zeitstempel im Journal. `processing task` (Start) bis zur ersten `find_slot: non-consecutive token position … with N new tokens` (Beginn des Bild-ubatch) = Encoder plus die 4 Text-Token davor. Von dort bis zur nächsten `find_slot`-Zeile = LLM-Prefill der Bild-Token.
  - Kontrolle mit Profil `bildtest-lv4` (`-lv 4`): `encoding mtmd batch` bis `decoding image batch` = Encoder, `image decoded (batch 1/1) in X ms` = LLM-Prefill.
  - Ergebnis: Beide Wege stimmen beim LLM-Anteil auf 0,03 s überein, beim Encoder liegt die Logdifferenz konstant 0,07 bis 0,08 s höher (die 4 Text-Token davor). Die Methode ohne Neustart ist damit belastbar. **gemessen**, `auswertung/lv4.md`.
- Zeit bis zur ersten Antwort = prompt_ms + Dauer des ersten Antwort-Tokens (30 bis 35 ms). Wanduhr des Clients liegt 0,1 bis 0,3 s darüber (HTTP, Bild dekodieren und skalieren im HTTP-Thread vor dem Slot).

## P1 Aufteilung (Profil coding, ohne Neustart)

**gemessen**, `raw/p1_split.jsonl`, je 3 Wiederholungen (2560: 2), Median (Spanne). Vollständig in `auswertung/p1.md`.

| Bild | Bild-Token | prompt_n | Encoder s | LLM-Prefill Bild s | Rest s | prompt_ms s | Encoder-Anteil |
|---|---|---|---|---|---|---|---|
| code 512 JPEG | 160 | 186 | 0,93 (0,92 bis 0,93) | 0,98 | 0,25 | 2,16 | 43 % |
| code 768 JPEG | 360 | 386 | 2,33 | 1,47 | 0,25 | 4,05 | 58 % |
| code 1024 JPEG | 640 | 666 | 4,99 | 1,81 | 0,25 | 7,04 | 71 % |
| code 1280 JPEG | 1000 | 1026 | 9,56 (9,53 bis 9,56) | 2,01 | 0,25 | 11,82 | 81 % |
| code 1280 PNG | 1000 | 1026 | 9,55 | 2,02 | 0,25 | 11,82 | 81 % |
| code 2560 PNG | 4000 | 4026 | 81,41 (81,37 bis 81,44) | 2,80 | 0,25 | 84,46 | 96 % |
| foto 768 JPEG | 360 | 386 | 2,36 | 1,67 | 0,25 | 4,27 | 55 % |
| foto 1280 JPEG | 1000 | 1026 | 9,54 | 2,06 | 0,25 | 11,85 | 81 % |
| tabelle 1280 JPEG | 1000 | 1026 | 9,62 | 2,04 | 0,25 | 11,91 | 81 % |

„Rest“ sind die 26 Text-Token nach dem Bild und das Sampling.

- **Tokenzahl:** Bild-Token = (B/32)·(H/32), also 512×320 → 160, 768×480 → 360, 1024×640 → 640, 1280×800 → 1000, 2560×1600 → 4000. **Kein Minimum von 1024 greift.** **gemessen**. Grenzen laut Startlog (`raw/journal_start_lv4.txt`): `image_min_pixels 8192`, `image_max_pixels 4194304`, also 8 bis 4096 Token bei 32×32 px je Token. Das sind die Code-Defaults, der mmproj trägt keine eigenen Grenzen (**gelesen**, `tools/mtmd/clip.cpp:1929` `set_limit_image_tokens(8, 4096)`, Schlüssel nur optional `clip.cpp:1931-1936`). Patch 16, Merge 2 im Startlog bestätigt.
- **Grounding-Warnung:** steht bei jedem Start im Journal („Qwen-VL models require at minimum 1024 image tokens … try adding --image-min-tokens 1024“), weil min_pixels unter 1024 Token liegt (**gelesen**, `clip.cpp:1938-1942`). Sie betrifft Bounding Boxes, nicht OCR (laut Issue 16842, nicht nachgeprüft).
- **Kappung auf n_ubatch:** greift bei diesem Modell nicht. Sie gilt nur für „mixed batch“ oder nicht-kausale Aufmerksamkeit (**gelesen**, `tools/server/server-context.cpp:1382-1392`, `4616-4618`). 4000 Token gingen ungekappt durch (**gemessen**).
- **Skalierung:** Encoder-Zeit je Token steigt von 5,8 ms (160 Token) auf 9,6 ms (1000) und 20,4 ms (4000). Potenzgesetz über alle Punkte: Token^1,40. Fit a·n + c·n² mit a = 5,69 ms, c = 3,67·10⁻³ ms trifft alle Punkte auf 8 %; bei 1000 Token ist der quadratische Teil 39 %, bei 4000 Token 72 % (Aufmerksamkeit über alle Patches). **gemessen** (Fit gerechnet), `auswertung/p1.md`.
- **LLM-Prefill der Bild-Token** hängt kaum an der Tokenzahl: 1,0 s für 160, 2,0 s für 1000, 2,8 s für 4000 Token. Das passt zu den festen Kosten je ubatch, die BENCHMARKS.md für die Experten auf der CPU misst (**vermutet**, nicht eigens nachgewiesen).
- **JPEG gegen PNG bei 1280:** gleich (9,56 gegen 9,55 s Encoder). Die Dekodierung liegt vor dem Slot und ist in prompt_ms nicht enthalten. **gemessen**.
- **Typ:** code, foto und tabelle bei 1280 innerhalb 0,1 s gleich. Der Encoder sieht nur die Pixelzahl. foto hat 0,2 s mehr LLM-Prefill bei 768 (1,67 gegen 1,47 s), Ursache offen. **gemessen**.
- **Vermutung „9 s Encoder + 3 s Prefill“:** für 1280 px fast richtig (9,5 + 2,0 + 0,25 s). Bei großen Bildern falsch: Dort ist es fast nur Encoder.

## P2 Qualität gegen Größe (ohne Neustart)

**gemessen**, `raw/p2_quality.jsonl`, je Bild eine Anfrage, Prompt „Transkribiere den gesamten sichtbaren Text exakt, Zeile für Zeile, ohne Kommentar.“, `max_tokens 3000`. Auswertung `scripts/qual.py`, Tabellen `auswertung/p2.md`.

| Bild | Token | prompt_ms s | code Probes / CER | fehler Probes / CER | tabelle Probes / CER | webseite Probes | Leiter Token richtig |
|---|---|---|---|---|---|---|---|
| 2560 PNG | 4000 | 84 | 7/9 / 0,5 % | 7/8 / 0,3 % | 10/10 / 10,0 % | 8/8 | 46/48 |
| 1280 PNG | 1000 | 11,9 | 6/9 / 1,4 % | | | | 31/48 |
| 1280 JPEG | 1000 | 11,9 | 6/9 / 1,7 % | 8/8 / 0,3 % | 10/10 / 2,9 % | 8/8 | 31/48 |
| 1024 JPEG | 640 | 7,2 | 6/9 / 1,0 % | 7/8 / 1,0 % | 10/10 / 2,9 % | 8/8 | 16/48 |
| 768 JPEG | 360 | 4,1 | 4/9 / 2,7 % | 4/8 / 29,9 % | 10/10 / 3,1 % | 8/8 | 7/48 |
| 512 JPEG | 160 | 2,2 | 0/9 / 523 % | 0/8 / 51,8 % | 9/10 / 6,9 % | 8/8 | 1/48 |
| Zuschnitt 1024 (unskaliert) | 640 | 7,2 | 5/6 / 0,9 % | | 2/2 | | |

CER global = Levenshtein über den ganzen Text gegen gt_lines, Leerraum normalisiert, bei code Zeilennummern und bei tabelle Spaltenstriche entfernt. Fenstertitel und andere Zusatzzeilen zählen als Fehler (deshalb tabelle 2560 bei 10 %: dort stand jede Zelle in einer eigenen Zeile, inhaltlich vollständig). Die reihenfolgefreie Zeilen-CER steht in `auswertung/p2.md`.

**Leiter, kleinste fehlerfrei gelesene Größe** (font_px_effective, von der größten Zeile abwärts ohne Fehler; in Klammern richtige Token des Blocks):

| Bild | A SF Pro hell | B SF Pro dunkel | C SF Mono hell | D SF Mono dunkel |
|---|---|---|---|---|
| 2560 PNG | 9 px (11/12) | 8 px (12/12) | 8 px (12/12) | 9 px (11/12) |
| 1280 PNG | 9 px (8/12) | 12 px (7/12) | 6,5 px (8/12) | 6 px (8/12) |
| 1280 JPEG | 9 px (8/12) | 12 px (8/12) | 6,5 px (8/12) | 8 px (7/12) |
| 1024 JPEG | 9,6 px (3/12) | 7,2 px (4/12) | 7,2 px (4/12) | 6,4 px (5/12) |
| 768 JPEG | 8,4 px (1/12) | 8,4 px (2/12) | 7,2 px (2/12) | 7,2 px (2/12) |
| 512 JPEG | keine (0/12) | 5,6 px (1/12) | keine (0/12) | keine (0/12) |

- **Kipppunkt:** Bis 1280 px liest das Modell Fließtext, Tabellen und Fehlermeldungen praktisch fehlerfrei (CER unter 2 %, alle Beträge und IDs). Zufällige Zeichenketten (Leiter-Token) brauchen etwa 9 bis 10 px effektive Schriftgröße für sichere Erkennung; darunter wird es lückenhaft, auch wenn einzelne 5,5-px-Zeilen noch stimmen. Ab 1024 px brechen die Leiter-Token ein (16 von 48), bei 768 auch die Fehlermeldung (Hex-Adressen, Crash-ID). **gemessen**.
- **Halluzination bei 512 px:** Bei code erfindet das Modell plausiblen Swift-Code und wiederholt Zeilen bis zu 2913 Antwort-Token; bei der Leiter erfindet es Token und Pixelangaben („A10 6px …“). Gefährlich, weil die Antwort glaubwürdig aussieht. **gemessen**, Antworttexte in der Rohdatei.
- **Verwechsler auch bei 2560:** `l1I|` und `"O0"` liest das Modell in keiner Größe richtig, `index_l1` erst bei 2560. **gemessen**.
- **Zuschnitt gegen verkleinertes Ganzes:** Der unskalierte 1024er-Ausschnitt von code hat die gleichen 640 Token wie das ganze Bild bei 1024 px, liest aber den sichtbaren Teil mit 0,9 % CER und 5/6 Probes (nur `"O0": 0` falsch, also so gut wie 2560). Das verkleinerte Ganze bei 1024 hat 6/9 Probes und verliert `index_l1`. Beim Tabellen-Ausschnitt sind beide Probes da. Ein Ausschnitt in voller Auflösung ist für Code also besser, deckt aber nur ein Viertel des Bildes ab. **gemessen**.
- **foto und diagramm** (Inhaltsfrage, Antworttexte in `auswertung/p2_inhalt.md`). **Urteil:** foto bei 2560 bis 768 richtig (abstrakte Wellen, rot bis lila und blau oben, grüne Hügel unten, kein Text). Bei 512 richtig im Kern, aber erfundene Zuordnung („iOS 16 Standard-Hintergrundgrafik“, tatsächlich macOS Sonoma). diagramm: alle sechs Kästen und alle sechs Pfeile in jeder Größe von 2560 bis 512 richtig. Die Kurvenwerte stimmen bis 768 ungefähr (Worker 380 bis 210 ms, Datenbank-Spitze im April); bei 512 vertauscht es die Serien und schreibt „2008“ statt 2026.

## P3 Präfix-Cache und Vorwärmen (ohne Neustart)

**gemessen**, `raw/p3_prewarm.jsonl`, Bild code-1280.jpg (1000 Token), je Szenario eigene Bildbytes, 1 bis 3 Wiederholungen. Alle Zeilen in `auswertung/p3.md`. Verlauf = 3 Text-Runden (165 Token).

| Szenario | Vorab-Anfrage | echte Anfrage: prompt_n / cache_n | echte Anfrage prompt_ms | Treffer |
|---|---|---|---|---|
| kalt, ohne Vorab (Bild + Frage) | – | 1028 / 0 | 11,85 s | – |
| gleiche Nachricht, Bild vor Text | user[Bild], max_tokens 1, 11,7 s | 1028 / 0 | 11,79 s | **nein** |
| gleiche Nachricht, Text vor Bild | user[Bild] | 1028 / 0 | 11,77 s | **nein** |
| gleiche Nachricht, Bild vor Text, mit Verlauf | Verlauf + user[Bild] | 1028 / 165 | 11,79 s | nur Verlauf |
| gleiche Nachricht, Text vor Bild, mit Verlauf | Verlauf + user[Bild] | 1028 / 165 | 11,80 s | nur Verlauf |
| **Bild als eigene Nachricht** | user[Bild], max_tokens 1 | 29 / 1014 | **0,38 s** (0,37 bis 0,39) | **ja** |
| eigene Nachricht, mit Verlauf | Verlauf + user[Bild] | 29 / 1179 | 0,37 s | ja |
| eigene Nachricht, Vorab mit `n_predict: 0` | wird angenommen, erzeugt 1 Token in 0,001 ms | 29 / 1014 | 0,38 s | ja |
| eigene Nachricht, Vorab streamend, Abbruch nach erstem Token | erstes Token nach 11,8 s | 33 / 1010 | 0,42 s | ja |
| gleiche Nachricht, `n_predict 0` bzw. streamend | | 1028 / 0 | 11,8 s | nein |

Die echte Anfrage beim Muster „eigene Nachricht“ ist `[user: Bild] [assistant: Antwort der Vorab-Anfrage] [user: Frage]`. Zeit bis zur ersten Antwort dort 0,4 s, Wanduhr 0,51 s.

- **Warum „gleiche Nachricht“ nie trifft (gelesen und mit `-lv 4` bestätigt):** Qwen3.6 ist hybrid, der rekurrente Zustand lässt sich nicht zurückrollen. Der Server braucht einen Checkpoint an einer Stelle, die höchstens so weit reicht wie der gemeinsame Präfix. Er legt Checkpoints an Anfängen von user-Nachrichten und 4 bzw. 4 + n_ubatch Token vor Promptende an (`server-context.cpp:4534`), aber **nie direkt nach einem Bild-Chunk** (`server-context.cpp:4590`: „do not checkpoint after mtmd chunks“). Die Vorab-Anfrage hat damit nur einen Checkpoint 4 Token vor ihrem Ende, also hinter der Stelle, an der die echte Anfrage abweicht (nach `<|vision_end|>`). Journal: „checking checkpoint with [49, 49] against 45 … forcing full prompt re-processing“ (`raw/p3_prewarm_lv4.jsonl`). Beim Muster „eigene Nachricht“ liegt dieser Checkpoint genau am Ende von `<|im_start|>assistant\n`, und das stimmt mit der echten Anfrage überein.
- **Fremde Textanfrage dazwischen** (anderes Gespräch, eigener Systemprompt): Das Bild-Präfix kommt aus dem RAM-Cache (`--cache-ram`) zurück. Echte Anfrage danach immer cache_n 1014, prompt_ms 0,37 s, Wanduhr 0,55 bis 0,72 s:

| Fremdanfrage | ihr prompt_ms | echte Anfrage danach: cache_n / prompt_ms / Wanduhr |
|---|---|---|
| 120 Token | 1,32 s | 1014 / 0,37 s / 0,56 s |
| 1900 Token | 2,39 s | 1014 / 0,38 s / 0,55 s |
| 7450 Token | 5,12 s | 1014 / 0,37 s / 0,59 s |
| 27 800 Token | 17,57 s | 1014 / 0,37 s / 0,72 s |
| 120 Token, Muster „gleiche Nachricht“ | 1,32 s | 0 / 11,78 s (kein Treffer, wie ohne Fremdanfrage) |

  Das Zurückholen kostet 0,05 bis 0,1 s („prompt cache update took 40 bis 78 ms“ im lv4-Journal). Eine Größengrenze habe ich bis 27 800 Token Fremdprompt nicht gefunden. Grenze ist der Platz: Ein Cache-Eintrag mit 1000-Token-Bild braucht 145 MiB, mit 4 Checkpoints 335 MiB, ein 7450-Token-Textprompt 397 MiB; bei 4096 MiB fliegt der älteste Eintrag raus („making room for prompt cache entry“). Nach etwa 10 bis 25 weiteren Gesprächen ist das Bild also weg (**vermutet** aus den Eintragsgrößen). **gemessen**.
- **Abbruch der Vorab-Anfrage nach 3 s** (Client schließt die Verbindung): Der Server loggt `cancel task` nach 3,0 s, rechnet aber Encoder und Bild-Prefill zu Ende. Slot frei erst nach 11,76 s, also genau so spät wie ohne Abbruch. Danach liegt das Bild verwertbar im Slot: echte Anfrage cache_n 1010, prompt_ms 0,40 s. Gleiches bei Abbruch nach 10 s (Slot frei nach 11,73 s). Ein Abbruch spart also keine Rechenzeit und gibt den Slot nicht früher frei, verwirft aber auch nichts. **gemessen**. Der Encoder läuft als ein Block im Worker-Thread (`server-context.cpp:4449-4451`), der Abbruch greift erst danach (**gelesen**, Zusammenhang **vermutet**).
- **Blockade:** Eine kurze fremde Textanfrage, 1 s nach der Vorab-Anfrage gestellt, braucht selbst 1,3 s prompt_ms, kommt aber erst nach 12,4 s Wanduhr zurück. Mit `-np 1` blockiert jede Bildanfrage den Slot für ihre volle Encoder-Zeit. **gemessen**.

## P4 Textpfad als Vergleich (ohne Neustart)

**gemessen**, `raw/p4_textpath.jsonl`, je 3 Wiederholungen, kalt (eindeutige Kennung im Anhang). Anhang als Codeblock in der user-Nachricht. „gt“ = `gt_lines` aus dem Manifest, „OCR“ = Apple-Vision-Text (accurate, ohne Sprachkorrektur, 2560er Bild, aus `mac-vision/ocr-text/`). Frage „Worum geht es in diesem Anhang? Ein Satz.“

| Typ | gt: prompt_n / bis 1. Token | OCR: prompt_n / bis 1. Token | gt + Bild 512 | OCR + Bild 512 | zum Vergleich Bild 1280 |
|---|---|---|---|---|---|
| code | 383 / 1,72 s | 411 / 1,74 s | 547 / 3,58 s | 573 / 3,60 s | 11,8 s |
| tabelle | 767 / 1,92 s | 845 / 1,97 s | 927 / 4,01 s | 1006 / 4,05 s | 11,9 s |
| fehler | 831 / 2,02 s | 846 / 2,03 s | 994 / 3,91 s | 1009 / 3,92 s | 11,8 s |
| webseite | 369 / 1,75 s | 366 / 1,74 s | 533 / 3,99 s | 528 / 3,96 s | 11,9 s |

- Text kostet 1,7 bis 2,0 s für 370 bis 850 Token. Davon ist der größte Teil fester Aufwand je ubatch, nicht die Tokenzahl (kurzer Text mit 120 Token: 1,3 s; 27 800 Token: 17,6 s, 1580 t/s, aus P3). **gemessen**.
- Das 512-px-Bild dazu kostet rund 2 s mehr (Encoder 0,9 s, eigener Bild-ubatch 1,0 s). **gemessen**.

### Zusatz: Antwortqualität mit OCR-Text statt Bild

**gemessen** (Zeiten, Antworten im Wortlaut in `raw/p4_ocrfragen.jsonl`), richtig/falsch ist **Urteil** gegen das Manifest. 20 Fragen, jede in drei Varianten, jede Anfrage kalt. Vollständige Tabelle mit Antworttexten in `auswertung/p4_ocrfragen.md`.

| Frage (Kurzform) | erwartet | (a) Bild 1280 | (b) nur OCR-Text | (c) OCR + Bild 512 |
|---|---|---|---|---|
| c1 Bezeichner der private(set)-Eigenschaft | `index_l1` | falsch `index_11` | falsch `index_11` | falsch `index_11` |
| c2 Hex-Werte im Vergleich | `0x7C`, `0x6C` | richtig | richtig | richtig |
| c3 Schlüssel mit Wert 0 | `"O0"` | falsch `"00"` | falsch („keinen“) | falsch `00` |
| c4 Rückgabetyp scanNext | `String?` | richtig | richtig | richtig |
| f1 Datei und Zeile:Spalte | `Parser.swift:214:17` | richtig | richtig | richtig |
| f2 fehlendes Member | `count_l` | richtig | richtig | richtig |
| f3 nie veränderte Variable | `idx_O` | falsch `idx_0` | falsch `idx_0` | falsch `idx_0` |
| f4 Exception Type und Crash-ID | `EXC_BAD_ACCESS (SIGSEGV)`, ID | richtig | richtig | richtig |
| t1 Stückpreis Gartenbank | 1.204,00 € | richtig | richtig | richtig |
| t2 Größe Gießkanne | 10 l | richtig | **falsch `101`** | **falsch `101`** |
| t3 Rabatt und Übertrag Kräuterregal | 12,5 %, 1.344,30 € | richtig | richtig | richtig |
| t4 Artikel aus Messing | Öllampe Messing | richtig | **falsch `Ollampe`** | **falsch `Ollampe`** |
| w1 Überschrift | Wie Stadtbäume Hitzesommer überstehen | richtig | richtig | richtig |
| w2 Prozent weniger Schäden | 18 % | richtig | richtig | richtig |
| w3 Balken 2023 | 17 | richtig | richtig | richtig |
| w4 Autorin und Datum | Mara Beispiel, 12. Oktober 2026 | richtig | richtig | richtig |
| d1 Kasten nach Job-Queue | Worker | richtig | **falsch** („keiner“) | richtig |
| d2 Ziele des API-Gateway | Auth-Dienst, Worker | richtig | **falsch** (Datenbank) | richtig |
| d3 wer zeigt auf Datenbank | Worker | richtig | **falsch** (Datenbank selbst) | richtig |
| d4 höchste Serie im April | Worker p95 | richtig | **falsch** (Gateway) | **falsch** (Datenbank) |
| **Summe richtig** | | **17/20** | **11/20** | **14/20** |
| **Zeit bis zur ersten Antwort**, Median (Spanne) | | 11,9 s (11,8 bis 12,3) | 1,8 s (1,3 bis 2,0) | 3,9 s (3,4 bis 4,1) |

- Je Typ (a/b/c): code 2/2/2, fehler 3/3/3, tabelle 4/2/2, webseite 4/4/4, diagramm 4/0/3.
- **Die OCR-Fehler schlagen voll durch** (t2, t4): Das Modell übernimmt `101` und `Ollampe` aus dem Text, auch wenn das 512-px-Bild daneben liegt. Bei 512 px kann es die Stelle nicht selbst prüfen. **Urteil**.
- **Ohne Bild fehlen Pfeile und Linien** (d1 bis d4): Das Modell rät aus der Textreihenfolge. Das 512er-Bild holt die Pfeile zurück (3/4), die Kurvenhöhe nicht. **Urteil**.
- `l`/`1` und `O`/`0` in Bezeichnern (c1, c3, f3) sind in allen drei Varianten falsch, auch mit dem 1280er-Bild. Das ist eine Grenze des Modells bei dieser Auflösung, nicht der OCR. **gemessen** (siehe auch P2).

## P5 Varianten mit Neustart

Je ein temporäres Profil `bildtest-<name>.env` = `coding` mit einer Änderung (`scripts/make_profiles.py`), Ausfälle in `ausfall.md`. Alle Profile sind wieder gelöscht.

### a) `--image-max-tokens` 256, 512, 768

**gemessen**, Latenz `raw/p5.jsonl` (1280: 3 Wiederholungen, 2560: 1), Qualität `raw/p5_quality.jsonl`. `--image-min-tokens` war nicht nötig (Default-Minimum 8 Token).

| Variante | Bild geschickt | Bild-Token | Encoder s | LLM-Bild s | prompt_ms s | Leiter richtig | Leiter kleinste fehlerfrei A/B/C/D, in px der 2560er Fassung | code Probes / CER | tabelle Probes / CER |
|---|---|---|---|---|---|---|---|---|---|
| coding | 1280 | 1000 | 9,56 | 2,01 | 11,82 | 31/48 | 18 / 24 / 13 / 16 | 6/9 / 1,7 % | 10/10 / 2,9 % |
| coding | 2560 | 4000 | 81,41 | 2,80 | 84,46 | 46/48 | 18 / 16 / 16 / 18 | 7/9 / 0,5 % | 10/10 / 10,0 % |
| imt768 | 1280 | 735 | 6,29 | 1,89 | 8,42 | 21/48 | 18 / 14 / 16 / 20 | 7/9 / 1,2 % | 10/10 / 3,5 % |
| imt768 | 2560 | 735 | 6,29 | 1,88 | 8,42 | 19/48 | 24 / 14 / 16 / 20 | 7/9 / 0,8 % | 10/10 / 3,5 % |
| imt512 | 1280 | 476 | 3,46 | 1,67 | 5,38 | 10/48 | 24 / 20 / 24 / 24 | 7/9 / 1,2 % | 10/10 / 0,6 % |
| imt512 | 2560 | 476 | 3,47 | 1,65 | 5,37 | 10/48 | 24 / 20 / 24 / 24 | 6/9 / 2,1 % | 10/10 / 3,5 % |
| imt256 | 1280 | 240 | 1,43 | 1,26 | 2,94 | 2/48 | keine / 28 / keine / 28 | 4/9 / 3,8 % | 9/10 / 6,7 % |
| imt256 | 2560 | 240 | 1,43 | 1,23 | 2,91 | 2/48 | keine / 28 / keine / 28 | 3/9 / 3,9 % | 9/10 / 5,7 % |

- **Der Server verkleinert zuverlässig:** 1280er und 2560er Original ergeben dieselbe Tokenzahl und dieselbe Zeit. Mit `--image-max-tokens 768` kostet das Retina-Original 8,4 s statt 84,5 s. Der Client kann das Original schicken. **gemessen**. Die Tokenzahl liegt etwas unter der Grenze (735 statt 768), weil die Seitenverhältnis-Rundung auf 32er-Raster abrundet (**vermutet**).
- **Qualität:** 768 Token sind für Code und Tabellen so gut wie 1000 (Probes gleich oder besser, CER 0,8 bis 3,5 %). Für kleine Schrift nicht: Leiter 21 statt 31 von 48. Bei 512 Token sind Tabellen noch gut, Leiter-Token fallen auf 10/48. 256 Token taugen nicht für OCR. Effektive Schriftgröße beim Server-Verkleinern ≈ px der 2560er Fassung × 0,44 (768), × 0,35 (512), × 0,25 (256) (**vermutet** aus den Tokenzahlen). **gemessen**.
- `--image-min-tokens` unter 1024 ist hier schon der Default (8). Für OCR wirkt nur die Pixelzahl, die Grounding-Warnung spielt keine Rolle (**vermutet**, Bounding Boxes nicht getestet).

### b) Threads `-t` 6, 10, 16, 20

**gemessen**, `raw/p5_1280.jsonl` (code-1280.jpg, 3 Wiederholungen), Text-Kontrolle `raw/tg.jsonl` (200 Token Antwort, 3 Wiederholungen). t10 frisch nach dem Neustart von `coding` gemessen.

| -t | Encoder s | LLM-Bild s | prompt_ms s | tg t/s Textbetrieb | prompt_ms kurzer Text |
|---|---|---|---|---|---|
| 6 | 14,59 | 2,02 | 16,91 | 34,9 (33,9 bis 35,0) | 0,88 s |
| **10** | **9,58** | 2,00 | **11,83** | 33,9 (33,6 bis 34,2) | 0,92 s |
| 16 | 10,09 | 2,04 | 12,37 | 36,5 | 0,89 s |
| 20 | 9,92 | 2,03 | 12,18 | 36,7 | 0,92 s |

- **10 Threads (= physische Kerne) sind für den Encoder am schnellsten.** 16 und 20 (Hyperthreads) sind 0,3 bis 0,5 s langsamer, 6 ist 5 s langsamer. Der Encoder bekommt `-t` (**gelesen**, `server-context.cpp:1251`, `clip.cpp:7530`).
- tg schwankt zwischen 33,9 und 36,7 t/s. Das liegt im bekannten Rauschen je Prozess (30 bis 37 t/s laut BENCHMARKS.md), ein Thread-Effekt auf tg ist so nicht belegbar. **gemessen**.

### c) Projektor Q8_0 auf der CPU gegen F16

Datei `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` aus `ggml-org/Qwen3.6-35B-A3B-GGUF@eb0b3cc97fecddab7eee6e6ffdbe92d2a00afcea`, 614 194 304 Byte, SHA256 `904cbf8c8e876220066ab3bf676c7efa40f3da372276fdaf8b01d2fb2a37a51d` (gegen HF-LFS geprüft), Eintrag in `~/models/SOURCES.txt`. Geladen nach dem Muster von `fetch.sh`, aber über `scripts/fetch_q8.py`, das die geschriebenen Seiten sofort aus dem Page-Cache wirft, weil nur 0,9 GB RAM wirklich frei waren und die gemappten Experten sonst verdrängt worden wären. Download 12:09:20 bis 12:09:39, während einer P2-Transkription (leiter-1280.jpg); deren Latenz wich nicht ab.

**gemessen**, `raw/p5_q8.jsonl`, `raw/p5_quality.jsonl`, `raw/procmem.jsonl`.

| | F16 (coding) | Q8_0 CPU |
|---|---|---|
| Encoder 512 / 768 / 1280 / 2560 | 0,93 / 2,33 / 9,56 / 81,4 s | 1,05 / 2,51 / 10,03 / 87,8 s |
| RAM anonym nach Start | 1272 MiB | 1000 MiB |
| Leiter 1280 / 768 richtig | 31/48 / 7/48 | 33/48 / 7/48 |
| code 1280 / 768 Probes | 6/9 / 4/9 | 6/9 / 4/9 |
| tabelle 1280 / 768 Probes | 10/10 / 10/10 | 10/10 / 10/10 |

- **Q8_0 ist auf der CPU 5 bis 8 % langsamer**, spart 270 MiB RAM, Qualität gleich (Leiter 33 statt 31 ist im Rauschen einer Einzelmessung). Auf der CPU lohnt Q8_0 nicht. **gemessen**.

### d) `--mtmd-batch-max-tokens`

Nicht gemessen, weil es nach Quelltext bei einem Bild je Anfrage nichts ändert. Die Grenze wird nur geprüft, wenn einem mtmd-Batch ein **weiteres** Bild hinzugefügt wird; das erste geht immer hinein (**gelesen**, `tools/mtmd/mtmd.cpp:3131-3146`). Sie wirkt also nur bei mehreren Bildern in einer Anfrage und nur, wenn der Projektor Batching kann (`clip_support_batch`, `clip.cpp:8077`). Auf Encoder-Speicher oder -Zeit eines einzelnen Bildes hat sie keinen Einfluss.

### e) Projektor auf der GPU (Regel 4: ein Versuch)

**Rechnung vorher** aus gemessenem VRAM:

- KV-Cache: 2560 MiB bei 131 072 Zellen, 10 Attention-Layer, f16, also **20 MiB je 1k Token**; rekurrenter Zustand 62,8 MiB fest; Modell 2039 MiB; CUDA-Rechenpuffer 2114 MiB bei 128k (**gemessen**, Startlog `raw/journal_start_lv4.txt`).
- Gesamt-VRAM je 1024 Token Kontext (KV plus mitwachsender Rechenpuffer): **28 MiB** (BENCHMARKS.md „Maximaler Kontext“: 7009 MiB bei 131 072, 7681 MiB bei 155 648 Token, ub 4096; 672 MiB auf 24 576 Token).
- Frei neben `coding` mit 128k: **831 MiB** (nvidia-smi 13:06, nicht 1183 wie in der Aufgabe angenommen).
- Projektor Q8_0: Gewichte 586 MiB, Rechenpuffer laut Startlog auf der CPU 248 MiB (Warmup mit 1472×1472 px = 2116 Token).

| Kontext | VRAM LLM (rechn.) | frei vor Projektor (rechn.) | frei nach Q8-Projektor (rechn., ~835 MiB) |
|---|---|---|---|
| 128k | 6966 MiB | 831 MiB | passt nicht (bekannter Absturz) |
| 96k | 6070 MiB | 1727 MiB | ~890 MiB |
| 64k | 5174 MiB | 2623 MiB | ~1790 MiB |
| 32k | 4278 MiB | 3519 MiB | ~2680 MiB |

Gewählt: **64k** (Regel 4 verlangt mindestens ~1,5 GB frei vor dem Projektor; 96k läge knapp darüber, ließe aber für ein 4000-Token-Bild wenig Reserve). Profil `bildtest-gpu-q8-64k` = `coding` mit `-c 65536`, Q8_0-Projektor und `--mmproj-offload`. Das sind drei Änderungen statt einer; ohne kleineren Kontext und Offload ist der Versuch nicht möglich.

**Ergebnis, gemessen** (`raw/p5_q8.jsonl`, `raw/procmem.jsonl`, `raw/gpu_q8_64k_nvsmi.csv`, `gpu("sample")` während der 1280er-Läufe):

- Start sauber nach 2,2 s, `NRestarts=0`, keine Fehlermeldung. VRAM nach Start 6010 MiB (Prozess), frei 1789 MiB. Vorhersage ~1790 MiB, gemessen 1789.
- VRAM-Spitze: 6173 MiB während der 1280er-Bilder (+122 MiB), **6573 MiB** nach dem 2560er-Bild (Rechenpuffer wächst über die Warmup-Größe, bleibt danach belegt). Minimum frei 1267 MiB.
- RAM anonym nach Start 410 MiB statt 1272 MiB.

| Bild | Encoder CPU F16 | Encoder GPU Q8_0 | prompt_ms CPU | prompt_ms GPU |
|---|---|---|---|---|
| 512 | 0,93 s | 0,11 s | 2,16 s | 1,35 s |
| 768 | 2,33 s | 0,18 s | 4,05 s | 1,90 s |
| 1280 | 9,56 s | 0,52 s | 11,82 s | **2,77 s** |
| 2560 | 81,41 s | 5,31 s | 84,46 s | **8,32 s** |

Encoder-Werte GPU nach der Logdifferenz-Methode, also einschließlich der 4 Text-Token davor (~0,07 s). Bei 512 px ist die erste Messung 0,23 s (Aufwärmen), danach 0,10 s.

- **Qualität** gleich wie Q8_0 auf der CPU: Leiter 1280 33/48, code 6/9, tabelle 10/10; Leiter 768 6/48 (CPU 7/48). **gemessen**.
- **Was es den Textbetrieb kostet:** Kontext 64k statt 128k. tg 36,3 t/s, kurzer Text-Prefill 0,91 s, beides wie mit 128k im Rauschen (**gemessen**, `raw/tg.jsonl`). Lange Prompts über 64k sind nicht möglich. Agent-Läufe mit 115k (BENCHMARKS.md Phase 6c) gingen damit nicht mehr. Zwischen 64k und 96k nicht gemessen.

## Was nicht gemessen wurde und warum

- `--mtmd-batch-max-tokens`: nach Quelltext ohne Wirkung bei einem Bild (P5d).
- GPU-Projektor mit F16 oder mit 96k: Regel 4 erlaubt einen Versuch; Q8_0 bei 64k war der sichere. F16 auf der GPU bei 64k ließe nach Rechnung ~1,5 GB frei (**vermutet**).
- Bounding Boxes und die Grounding-Warnung: nicht Teil der Aufgabe.
- Mehrere Bilder je Anfrage, echte Fotos mit Text, Bilder im Verlauf nach Verdrängung aus dem RAM-Cache (wie viele Gespräche genau): nicht gemessen.
- Einzelmessungen bei Qualität (temperature 0): keine Wiederholungen, Schwankung zwischen Prozessen unbekannt.

## Überraschungen gegenüber der Ausgangsvermutung (9 s Encoder + 3 s Prefill)

1. Bei 1280 px stimmt die Größenordnung (9,5 + 2,0 s). Bei 2560 px nicht: 81 s Encoder, nur 2,8 s LLM. Der LLM-Anteil ist fast konstant, der Encoder wächst überlinear.
2. Auf der GPU ist der Encoder 13- bis 18-mal schneller. Bei 1280 px bleibt dann der LLM-Prefill (2,0 s) der größte Posten.
3. Der Prompt-Cache hilft beim Vorwärmen nur mit „Bild als eigene Nachricht“. Bild und Frage in einer Nachricht treffen nie, auch nicht mit Verlauf oder anderer Reihenfolge.
4. Ein abgebrochener Vorab-Request hält den Slot trotzdem bis zum Ende fest, legt das Bild aber verwertbar in den Cache.
5. Das Modell liest `l`/`1`/`I` und `O`/`0` in Bezeichnern auch bei 1280 px falsch, genau wie Apple Vision.
6. Bei 512 px erfindet das Modell Code, statt „unlesbar“ zu sagen.
7. Der RAM-Prompt-Cache füllt sich mit Bildern schnell: Ein Eintrag mit 1000-Token-Bild ist 145 bis 335 MiB groß. Nach P1 bis P4 lag der Dienst bei 7010 MiB anonymem RAM, nur 0,9 GB waren wirklich frei (`raw/procmem.jsonl`, engpass `ram`).
