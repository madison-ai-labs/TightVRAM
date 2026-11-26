# Lauf gestört — nicht werten

Während dieses Laufs lief auf der Testmaschine parallel eine SWE-bench-Suite einer anderen Session
(podman, `--cpus 8`, Django-Tests `--parallel 8`). Container laut journal:
17:07:06–17:10:52 (django-14122), 17:12:44–17:16:21 (django-14915).

Betroffen: p0–p3 (Kalibrierung komplett), p4 A, p6 S, p7 S, Anfang p8 P.
Sauber: p5 P (bis auf die letzten ~6 s), p9–p14.

Ursache: Die Warte-Prüfung vor dem Start suchte `swebench-fullsuite/driver.sh`, der Treiber lief
aber als `/bin/bash ./driver.sh` (von der anderen Session neu gestartet), und die beiden Prüfungen
um 17:06:20/17:06:50 fielen genau in die Pause zwischen zwei Containern (17:06:12–17:07:06).

Die Urteile in erg3.json (Q3, MTP, F) gelten deshalb nicht. Wiederholung als eigener Lauf.
