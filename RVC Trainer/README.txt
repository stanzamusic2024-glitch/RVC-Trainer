RVC Trainer - Trainingsprogramm fuer RVC (Retrieval-based Voice Conversion)
===========================================================================

Installation / Voraussetzungen:
- Bestehende RVC-beta0717 Installation (Standard: F:\Vocal App\RVC-beta0717)
- Python 3.9 Runtime in der RVC-Installation (runtime\python.exe)
- NVIDIA GPU mit CUDA (empfohlen) oder CPU

Start:
- start.bat doppelklicken, oder:
- python rvc_trainer.py

Verwendung:
1. Tab "Einstellungen": RVC-Home und Python-Runtime pruefen/anpassen, speichern.
2. Tab "Training":
   - WAV-Datei auswaehlen (Stimmaufnahme, moeglichst sauber, 10-30 Min)
   - Modellname vergeben
   - Sample Rate (32k/40k/48k), Version (v1/v2), F0-Methode waehlen
   - Epochs (z.B. 200-500), Batch Size (z.B. 4) einstellen
   - Export-Ordner waehlen
   - "Training starten" klicken
3. Tab "Logs": Fortschritt verfolgen.
4. Nach Abschluss: {Modellname}.pth und {Modellname}.index im Export-Ordner.

Hinweise:
- Training kann je nach Datenmenge und Epochs mehrere Stunden dauern.
- GPU mit >= 6 GB VRAM empfohlen (fp16).
- Bei Problemen: Logs pruefen, RVC-Home in Einstellungen korrigieren.

Dateien:
- rvc_trainer.py         : Hauptprogramm (Tkinter GUI)
- train_index_helper.py   : Hilfsskript fuer FAISS-Index-Training
- start.bat               : Launcher
- rvc_trainer_config.json : Gespeicherte Einstellungen (wird automatisch erstellt)
