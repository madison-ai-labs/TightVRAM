### Kontrolle der Aufteilung mit -lv 4 (raw/lv4.jsonl)

| Bild | n | Encoder s (lv4: 'encoding mtmd batch' bis 'decoding image batch') | Encoder s (Logdifferenz-Methode) | LLM-Bild s (lv4: 'image decoded in') | LLM-Bild s (Logdifferenz) | prompt_ms s |
|---|---|---|---|---|---|---|
| code-512.jpg | 3 | 0,85 (0,85–0,86) | 0,93 (0,92–1,00) | 0,99 (0,98–1,01) | 0,99 (0,98–1,01) | 2,17 (2,16–2,26) |
| code-768.jpg | 3 | 2,27 (2,27–2,28) | 2,35 (2,34–2,35) | 1,48 (1,47–1,48) | 1,47 (1,47–1,48) | 4,07 (4,06–4,07) |
| code-1024.jpg | 3 | 5,00 (5,00–5,01) | 5,07 (5,07–5,08) | 1,81 (1,81–1,82) | 1,81 (1,81–1,81) | 7,13 (7,13–7,14) |
| code-1280.jpg | 3 | 9,51 (9,51–9,51) | 9,58 (9,58–9,59) | 2,02 (2,02–2,02) | 2,02 (2,01–2,02) | 11,85 (11,84–11,85) |
| code-2560.png | 1 | 81,15 | 81,26 | 2,83 | 2,80 | 84,30 |
