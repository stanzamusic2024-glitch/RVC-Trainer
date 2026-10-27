# -*- coding: utf-8 -*-
"""RVC Trainer - Vereinfachtes Trainingsprogramm fuer RVC (Retrieval-based Voice Conversion)."""
import os, sys, subprocess, threading, shutil, time, json, re

DEFAULT_RVC_HOME = r"F:\Vocal App\RVC-beta0717"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_trainer_config.json")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

APP_TITLE = "RVC Trainer"


def load_config():
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    cfg.setdefault("rvc_home", DEFAULT_RVC_HOME)
    cfg.setdefault("python_cmd", None)
    cfg.setdefault("gpus", "0")
    cfg.setdefault("n_cpu", 4)
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


class RvcTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("920x760")

        self.cfg = load_config()
        self.rvc_home = self.cfg.get("rvc_home", DEFAULT_RVC_HOME)
        self.python_cmd = self.cfg.get("python_cmd")
        if not self.python_cmd:
            self.python_cmd = os.path.join(self.rvc_home, "runtime", "python.exe")
        self.process = None
        self.stop_flag = False
        self.running = False

        self.total_epochs = 200
        self.progress_var = tk.DoubleVar(value=0.0)
        self.console_win = None
        self.console_text = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._log("RVC Trainer gestartet.")
        self._log("RVC-Home: %s" % self.rvc_home)
        self._log("Python: %s" % self.python_cmd)

    # ---------- UI ----------
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab_train = ttk.Frame(nb)
        nb.add(tab_train, text="Training")
        self._build_train_tab(tab_train)

        tab_log = ttk.Frame(nb)
        nb.add(tab_log, text="Logs")
        self._build_log_tab(tab_log)

        tab_set = ttk.Frame(nb)
        nb.add(tab_set, text="Einstellungen")
        self._build_settings_tab(tab_set)

        self.status_var = tk.StringVar(value="Bereit")
        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill="x", side="bottom")

    def _build_train_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Trainingsparameter", padding=10)
        frm.pack(fill="x", padx=10, pady=10)

        ttk.Label(frm, text="WAV-Datei:").grid(row=0, column=0, sticky="w", pady=4)
        self.wav_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.wav_var, width=60).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="Durchsuchen...", command=self._pick_wav).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(frm, text="Modellname:").grid(row=1, column=0, sticky="w", pady=4)
        self.name_var = tk.StringVar(value="mein_model")
        ttk.Entry(frm, textvariable=self.name_var, width=60).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="Sample Rate:").grid(row=2, column=0, sticky="w", pady=4)
        self.sr_var = tk.StringVar(value="40k")
        ttk.Combobox(frm, textvariable=self.sr_var, values=["32k", "40k", "48k"], state="readonly", width=10).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Version:").grid(row=3, column=0, sticky="w", pady=4)
        self.ver_var = tk.StringVar(value="v2")
        ttk.Combobox(frm, textvariable=self.ver_var, values=["v1", "v2"], state="readonly", width=10).grid(row=3, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="F0-Methode:").grid(row=4, column=0, sticky="w", pady=4)
        self.f0_var = tk.StringVar(value="rmvpe")
        ttk.Combobox(frm, textvariable=self.f0_var, values=["pm", "harvest", "crepe", "rmvpe"], state="readonly", width=10).grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Epochs (total):").grid(row=5, column=0, sticky="w", pady=4)
        self.epoch_var = tk.StringVar(value="200")
        ttk.Entry(frm, textvariable=self.epoch_var, width=12).grid(row=5, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Speichern alle X Epochs:").grid(row=6, column=0, sticky="w", pady=4)
        self.saveepoch_var = tk.StringVar(value="50")
        ttk.Entry(frm, textvariable=self.saveepoch_var, width=12).grid(row=6, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Batch Size:").grid(row=7, column=0, sticky="w", pady=4)
        self.batch_var = tk.StringVar(value="4")
        ttk.Entry(frm, textvariable=self.batch_var, width=12).grid(row=7, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="F0 verwenden:").grid(row=8, column=0, sticky="w", pady=4)
        self.if_f0_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, variable=self.if_f0_var).grid(row=8, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Cache in GPU:").grid(row=9, column=0, sticky="w", pady=4)
        self.cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, variable=self.cache_var).grid(row=9, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Jede Speicherung als .pth:").grid(row=10, column=0, sticky="w", pady=4)
        self.saveweights_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, variable=self.saveweights_var).grid(row=10, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Export-Ordner:").grid(row=11, column=0, sticky="w", pady=4)
        self.outdir_var = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "export"))
        ttk.Entry(frm, textvariable=self.outdir_var, width=60).grid(row=11, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="Durchsuchen...", command=self._pick_outdir).grid(row=11, column=2, padx=4, pady=4)

        frm.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(btn_frame, text="Training starten", command=self._start_training)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btn_frame, text="Stoppen", command=self._stop_training, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.console_btn = ttk.Button(btn_frame, text="Konsole anzeigen", command=self._open_console)
        self.console_btn.pack(side="left", padx=4)

        # Fortschrittsbalken
        prog_frame = ttk.Frame(parent)
        prog_frame.pack(fill="x", padx=10, pady=5)
        self.progress = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.pct_label = ttk.Label(prog_frame, text="0 %")
        self.pct_label.pack(side="left")

        self.train_status = tk.StringVar(value="Bereit zum Trainieren.")
        ttk.Label(parent, textvariable=self.train_status).pack(padx=10, pady=5, anchor="w")

    def _build_log_tab(self, parent):
        self.log_text = scrolledtext.ScrolledText(parent, wrap="word", height=30)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.configure(state="disabled")

    def _build_settings_tab(self, parent):
        frm = ttk.LabelFrame(parent, text="Pfade & System", padding=10)
        frm.pack(fill="x", padx=10, pady=10)

        ttk.Label(frm, text="RVC-Home:").grid(row=0, column=0, sticky="w", pady=4)
        self.rvc_var = tk.StringVar(value=self.rvc_home)
        ttk.Entry(frm, textvariable=self.rvc_var, width=60).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="Durchsuchen...", command=self._pick_rvc).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(frm, text="Python (RVC-Runtime):").grid(row=1, column=0, sticky="w", pady=4)
        self.py_var = tk.StringVar(value=self.python_cmd)
        ttk.Entry(frm, textvariable=self.py_var, width=60).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="Durchsuchen...", command=self._pick_py).grid(row=1, column=2, padx=4, pady=4)

        ttk.Label(frm, text="GPUs (z.B. 0 oder 0-1):").grid(row=2, column=0, sticky="w", pady=4)
        self.gpu_var = tk.StringVar(value=self.cfg.get("gpus", "0"))
        ttk.Entry(frm, textvariable=self.gpu_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="CPU-Kerne:").grid(row=3, column=0, sticky="w", pady=4)
        self.cpu_var = tk.StringVar(value=str(self.cfg.get("n_cpu", 4)))
        ttk.Entry(frm, textvariable=self.cpu_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

        frm.columnconfigure(1, weight=1)

        btn_save = ttk.Button(frm, text="Einstellungen speichern", command=self._save_settings)
        btn_save.grid(row=4, column=0, columnspan=3, pady=10, sticky="w")

        info = ttk.LabelFrame(parent, text="Info", padding=10)
        info.pack(fill="x", padx=10, pady=10)
        ttk.Label(info, text="Stellt sicher, dass die RVC-beta0717 Installation vollstaendig ist.\n"
                             "Das Programm nutzt die vorhandene RVC-Runtime (Python 3.9 + torch).\n"
                             "Training kann je nach Datenmenge und Epochs mehrere Stunden dauern.").pack(anchor="w")

    # ---------- Helpers ----------
    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = "[%s] %s\n" % (ts, msg)
        # Thread-safe GUI-Updates via root.after
        def _update():
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line)
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except Exception:
                pass
            try:
                if self.console_text is not None:
                    self.console_text.configure(state="normal")
                    self.console_text.insert("end", line)
                    self.console_text.see("end")
                    self.console_text.configure(state="disabled")
            except Exception:
                pass
        try:
            self.root.after(0, _update)
        except Exception:
            pass
        print(line, end="", flush=True)

    def _set_progress(self, pct):
        pct = max(0.0, min(100.0, pct))
        def _update():
            try:
                self.progress_var.set(pct)
                self.pct_label.config(text="%d %%" % int(pct))
            except Exception:
                pass
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def _open_console(self):
        if self.console_win is not None and self.console_win.winfo_exists():
            self.console_win.lift()
            return
        self.console_win = tk.Toplevel(self.root)
        self.console_win.title("RVC Trainer - Konsole")
        self.console_win.geometry("800x400")
        self.console_text = scrolledtext.ScrolledText(
            self.console_win, wrap="word", bg="black", fg="#00ff00",
            insertbackground="#00ff00", font=("Consolas", 10))
        self.console_text.pack(fill="both", expand=True)
        self.console_text.configure(state="disabled")

    def _set_status(self, msg):
        def _update():
            try:
                self.status_var.set(msg)
            except Exception:
                pass
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def _pick_wav(self):
        p = filedialog.askopenfilename(title="WAV-Datei auswaehlen", filetypes=[("WAV", "*.wav"), ("Alle", "*.*")])
        if p:
            self.wav_var.set(p)

    def _pick_outdir(self):
        p = filedialog.askdirectory(title="Export-Ordner auswaehlen")
        if p:
            self.outdir_var.set(p)

    def _pick_rvc(self):
        p = filedialog.askdirectory(title="RVC-Home auswaehlen")
        if p:
            self.rvc_var.set(p)

    def _pick_py(self):
        p = filedialog.askopenfilename(title="Python-Runtime auswaehlen", filetypes=[("Python", "python.exe"), ("Alle", "*.*")])
        if p:
            self.py_var.set(p)

    def _save_settings(self):
        self.rvc_home = self.rvc_var.get().strip()
        self.python_cmd = self.py_var.get().strip()
        self.cfg["rvc_home"] = self.rvc_home
        self.cfg["python_cmd"] = self.python_cmd
        self.cfg["gpus"] = self.gpu_var.get().strip()
        try:
            self.cfg["n_cpu"] = int(self.cpu_var.get().strip())
        except ValueError:
            self.cfg["n_cpu"] = 4
        save_config(self.cfg)
        self._log("Einstellungen gespeichert.")
        messagebox.showinfo("Info", "Einstellungen gespeichert.")

    def _run_subprocess(self, cmd, cwd=None, log_output=True, progress_range=None):
        """Fuehrt einen Befehl aus und streamt stdout/stderr in das Log.

        progress_range: (start_pct, end_pct) fuer den Fortschrittsbalken.
        """
        self._log("CMD: %s" % cmd)
        if cwd is None:
            cwd = self.rvc_home
        try:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except Exception as e:
            self._log("Fehler beim Starten: %s" % e)
            return False
        self.process = proc
        start_pct, end_pct = (progress_range if progress_range else (None, None))
        try:
            for line in proc.stdout:
                if self.stop_flag:
                    proc.terminate()
                    break
                line = line.rstrip("\n")
                if log_output and line.strip():
                    self._log(line)
                    # Epoch-Parsing fuer Fortschritt
                    if progress_range and self.total_epochs > 0:
                        m = re.search(r"Epoch[:\s]+(\d+)", line)
                        if m:
                            ep = int(m.group(1))
                            frac = min(1.0, ep / max(1, self.total_epochs))
                            pct = start_pct + (end_pct - start_pct) * frac
                            self._set_progress(pct)
                if progress_range and start_pct is not None and end_pct is not None:
                    # leichte Progression auch ohne Epoch-Match
                    pass
        except Exception:
            pass
        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        self.process = None
        if self.stop_flag:
            self._log("Prozess abgebrochen.")
            return False
        if proc.returncode != 0:
            self._log("Prozess beendet mit Code %d" % proc.returncode)
            return False
        self._log("Prozess erfolgreich beendet.")
        if progress_range:
            self._set_progress(end_pct)
        return True

    # ---------- Training Pipeline ----------
    def _start_training(self):
        if self.running:
            messagebox.showwarning("Laeuft", "Ein Training laeuft bereits.")
            return
        wav = self.wav_var.get().strip()
        name = self.name_var.get().strip()
        if not wav or not os.path.exists(wav):
            messagebox.showerror("Fehler", "Bitte eine gueltige WAV-Datei auswaehlen.")
            return
        if not name:
            messagebox.showerror("Fehler", "Bitte einen Modellnamen angeben.")
            return
        name = "".join(c for c in name if c.isalnum() or c in "_-")
        if not name:
            messagebox.showerror("Fehler", "Ungueltiger Modellname.")
            return
        self.name_var.set(name)

        try:
            self.total_epochs = int(self.epoch_var.get().strip())
        except ValueError:
            self.total_epochs = 200

        # Konsole-Fenster oeffnen
        self._open_console()

        self.stop_flag = False
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._set_progress(0.0)
        self.train_status.set("Training wird vorbereitet...")
        self._set_status("Training laeuft...")

        t = threading.Thread(target=self._training_worker, args=(wav, name), daemon=True)
        t.start()

    def _stop_training(self):
        self.stop_flag = True
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass
        self._log("Stop angefordert.")

    def _on_close(self):
        self.stop_flag = True
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass
        if self.console_win is not None:
            try:
                self.console_win.destroy()
            except Exception:
                pass
        self.root.destroy()

    def _finish_training(self, success, name):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if success:
            self._set_progress(100.0)
            self.train_status.set("Training abgeschlossen: %s" % name)
            self._set_status("Fertig")
            messagebox.showinfo("Fertig", "Training abgeschlossen.\nModell: %s" % name)
        else:
            self.train_status.set("Training fehlgeschlagen oder abgebrochen.")
            self._set_status("Fehler/Abbruch")

    def _training_worker(self, wav_path, name):
        try:
            success = self._run_pipeline(wav_path, name)
        except Exception as e:
            self._log("Unerwarteter Fehler: %s" % e)
            import traceback
            self._log(traceback.format_exc())
            success = False
        self.root.after(0, lambda: self._finish_training(success, name))

    def _run_pipeline(self, wav_path, name):
        rvc = self.rvc_home
        py = '"%s"' % self.python_cmd
        sr_str = self.sr_var.get().strip()
        version = self.ver_var.get().strip()
        f0method = self.f0_var.get().strip()
        total_epoch = int(self.epoch_var.get().strip())
        save_epoch = int(self.saveepoch_var.get().strip())
        batch_size = int(self.batch_var.get().strip())
        if_f0 = 1 if self.if_f0_var.get() else 0
        gpus = self.gpu_var.get().strip() or self.cfg.get("gpus", "0")
        n_cpu = int(self.cpu_var.get().strip() or self.cfg.get("n_cpu", 4))
        cache_gpu = 1 if self.cache_var.get() else 0
        save_weights = 1 if self.saveweights_var.get() else 0
        sr_num = {"32k": 32000, "40k": 40000, "48k": 48000}[sr_str]
        echl = 128

        if version == "v2":
            pre_dir = os.path.join(rvc, "pretrained_v2")
        else:
            pre_dir = os.path.join(rvc, "pretrained")
        f0_prefix = "f0" if if_f0 else ""
        preG = os.path.join(pre_dir, "%sG%s.pth" % (f0_prefix, sr_str))
        preD = os.path.join(pre_dir, "%sD%s.pth" % (f0_prefix, sr_str))
        if not os.path.exists(preG):
            preG = ""
        if not os.path.exists(preD):
            preD = ""

        exp_dir_abs = os.path.join(rvc, "logs", name)
        os.makedirs(exp_dir_abs, exist_ok=True)

        trainset_dir = os.path.join(rvc, "trainset_tmp", name)
        if os.path.exists(trainset_dir):
            shutil.rmtree(trainset_dir, ignore_errors=True)
        os.makedirs(trainset_dir, exist_ok=True)
        shutil.copy2(wav_path, os.path.join(trainset_dir, os.path.basename(wav_path)))
        self._log("Trainset-Verzeichnis: %s" % trainset_dir)

        os.makedirs(exp_dir_abs, exist_ok=True)
        for lf in ["preprocess.log", "extract_f0_feature.log"]:
            p = os.path.join(exp_dir_abs, lf)
            try:
                open(p, "w").close()
            except Exception:
                pass

        csvdb_dir = os.path.join(rvc, "csvdb")
        os.makedirs(csvdb_dir, exist_ok=True)
        try:
            with open(os.path.join(csvdb_dir, "stop.csv"), "w", newline="") as f:
                f.write("False")
        except Exception:
            pass

        # ---------- Schritt 1: Preprocess (0-10%) ----------
        self.root.after(0, lambda: self.train_status.set("Schritt 1/6: Audio vorbereiten..."))
        self._log("=== Schritt 1: Preprocess ===")
        self._set_progress(2.0)
        cmd = '%s trainset_preprocess_pipeline_print.py "%s" %d %d "%s/logs/%s" False' % (
            py, trainset_dir, sr_num, n_cpu, rvc, name
        )
        ok = self._run_subprocess(cmd, cwd=rvc, progress_range=(2.0, 10.0))
        if not ok or self.stop_flag:
            return False

        # ---------- Schritt 2: F0 Extraction (10-20%) ----------
        if if_f0:
            self.root.after(0, lambda: self.train_status.set("Schritt 2/6: F0 extrahieren..."))
            self._log("=== Schritt 2: F0 Extraction ===")
            cmd = '%s extract_f0_print.py "%s/logs/%s" %d %s %d' % (
                py, rvc, name, n_cpu, f0method, echl
            )
            ok = self._run_subprocess(cmd, cwd=rvc, progress_range=(10.0, 20.0))
            if not ok or self.stop_flag:
                return False

        # ---------- Schritt 3: Feature Extraction (20-30%) ----------
        self.root.after(0, lambda: self.train_status.set("Schritt 3/6: Features extrahieren..."))
        self._log("=== Schritt 3: Feature Extraction ===")
        gpus_list = gpus.split("-")
        leng = len(gpus_list)
        procs = []
        for idx, g in enumerate(gpus_list):
            cmd = '%s extract_feature_print.py %s %d %d %s "%s/logs/%s" %s' % (
                py, "cuda:0", leng, idx, g, rvc, name, version
            )
            self._log("CMD: %s" % cmd)
            proc = subprocess.Popen(cmd, shell=True, cwd=rvc, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            procs.append(proc)
        import threading as _th
        def _stream(p):
            for line in p.stdout:
                line = line.rstrip("\n")
                if line.strip():
                    self._log(line)
        threads = []
        for p in procs:
            t = _th.Thread(target=_stream, args=(p,), daemon=True)
            t.start()
            threads.append(t)
        for p in procs:
            p.wait()
        for t in threads:
            t.join()
        for p in procs:
            if p.returncode != 0:
                self._log("Feature-Extraktion fehlgeschlagen (code %d)" % p.returncode)
                return False
        if self.stop_flag:
            return False
        self._set_progress(30.0)

        # ---------- Schritt 3b: filelist.txt erstellen ----------
        self._log("Erstelle filelist.txt...")
        gt_wavs_dir = os.path.join(exp_dir_abs, "0_gt_wavs")
        feature_dir = os.path.join(exp_dir_abs, "3_feature256" if version == "v1" else "3_feature768")
        f0_dir = os.path.join(exp_dir_abs, "2a_f0")
        f0nsf_dir = os.path.join(exp_dir_abs, "2b-f0nsf")
        spk_id = 0

        if if_f0:
            names = (
                set(n.split(".")[0] for n in os.listdir(gt_wavs_dir))
                & set(n.split(".")[0] for n in os.listdir(feature_dir))
                & set(n.split(".")[0] for n in os.listdir(f0_dir))
                & set(n.split(".")[0] for n in os.listdir(f0nsf_dir))
            )
        else:
            names = set(n.split(".")[0] for n in os.listdir(gt_wavs_dir)) & set(
                n.split(".")[0] for n in os.listdir(feature_dir)
            )

        opt = []
        for n in names:
            if if_f0:
                opt.append("%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s" % (
                    gt_wavs_dir.replace("\\", "\\\\"), n,
                    feature_dir.replace("\\", "\\\\"), n,
                    f0_dir.replace("\\", "\\\\"), n,
                    f0nsf_dir.replace("\\", "\\\\"), n,
                    spk_id,
                ))
            else:
                opt.append("%s/%s.wav|%s/%s.npy|%s" % (
                    gt_wavs_dir.replace("\\", "\\\\"), n,
                    feature_dir.replace("\\", "\\\\"), n,
                    spk_id,
                ))
        fea_dim = 256 if version == "v1" else 768
        if if_f0:
            for _ in range(2):
                opt.append("%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s" % (
                    rvc, sr_str, rvc, fea_dim, rvc, rvc, spk_id))
        else:
            for _ in range(2):
                opt.append("%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s" % (
                    rvc, sr_str, rvc, fea_dim, spk_id))
        import random
        random.shuffle(opt)
        with open(os.path.join(exp_dir_abs, "filelist.txt"), "w") as f:
            f.write("\n".join(opt))
        self._log("filelist.txt erstellt (%d Eintraege)." % len(opt))

        wav16_dir = os.path.join(exp_dir_abs, "1_16k_wavs")
        log_interval = 1
        if os.path.exists(wav16_dir):
            wavs = [f for f in os.listdir(wav16_dir) if f.endswith(".wav")]
            if wavs:
                log_interval = -(-len(wavs) // batch_size)
                if log_interval > 1:
                    log_interval += 1
        self._log("log_interval = %d" % log_interval)

        # ---------- Schritt 4: Training (30-90%) ----------
        self.root.after(0, lambda: self.train_status.set("Schritt 4/6: Training..."))
        self._log("=== Schritt 4: Training ===")
        if gpus:
            cmd = (
                '%s train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %d -bs %d -g %s -te %d -se %d %s %s -l %d -c %d -sw %d -v %s -li %d'
                % (
                    py, name, sr_str, if_f0, batch_size, gpus, total_epoch, save_epoch,
                    ("-pg \"%s\"" % preG) if preG else "",
                    ("-pd \"%s\"" % preD) if preD else "",
                    1, cache_gpu, save_weights, version, log_interval,
                )
            )
        else:
            cmd = (
                '%s train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %d -bs %d -te %d -se %d %s %s -l %d -c %d -sw %d -v %s -li %d'
                % (
                    py, name, sr_str, if_f0, batch_size, total_epoch, save_epoch,
                    ('-pg "%s"' % preG) if preG else "\b",
                    ('-pd "%s"' % preD) if preD else "\b",
                    1, cache_gpu, save_weights, version, log_interval,
                )
            )
        ok = self._run_subprocess(cmd, cwd=rvc, progress_range=(30.0, 90.0))
        if not ok or self.stop_flag:
            return False

        # ---------- Schritt 5: Index Training (90-95%) ----------
        self.root.after(0, lambda: self.train_status.set("Schritt 5/6: Index trainieren..."))
        self._log("=== Schritt 5: Index Training ===")
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_index_helper.py")
        cmd = '%s "%s" %s %s "%s"' % (py, helper, name, version, rvc)
        ok = self._run_subprocess(cmd, cwd=rvc, progress_range=(90.0, 95.0))
        if not ok or self.stop_flag:
            return False

        # ---------- Schritt 6: Export (95-100%) ----------
        self.root.after(0, lambda: self.train_status.set("Schritt 6/6: Exportieren..."))
        self._log("=== Schritt 6: Export ===")
        self._set_progress(96.0)
        outdir = self.outdir_var.get().strip()
        os.makedirs(outdir, exist_ok=True)

        pth_candidates = []
        weights_dir = os.path.join(rvc, "weights")
        exact = os.path.join(weights_dir, "%s.pth" % name)
        if os.path.exists(exact):
            pth_candidates.append(exact)
        else:
            for fn in os.listdir(weights_dir):
                if fn.startswith(name) and fn.endswith(".pth"):
                    pth_candidates.append(os.path.join(weights_dir, fn))
            if pth_candidates:
                pth_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        index_candidates = []
        for fn in os.listdir(exp_dir_abs):
            if fn.endswith(".index") and "added" in fn:
                index_candidates.append(os.path.join(exp_dir_abs, fn))
        if not index_candidates:
            for fn in os.listdir(exp_dir_abs):
                if fn.endswith(".index"):
                    index_candidates.append(os.path.join(exp_dir_abs, fn))

        if not pth_candidates:
            self._log("Kein .pth gefunden in %s" % weights_dir)
            return False
        if not index_candidates:
            self._log("Kein .index gefunden in %s" % exp_dir_abs)
            return False

        pth_src = pth_candidates[0]
        index_src = index_candidates[0]
        pth_dst = os.path.join(outdir, "%s.pth" % name)
        index_dst = os.path.join(outdir, "%s.index" % name)
        shutil.copy2(pth_src, pth_dst)
        shutil.copy2(index_src, index_dst)
        self._log("Exportiert: %s" % pth_dst)
        self._log("Exportiert: %s" % index_dst)
        self._set_progress(100.0)

        return True


def main():
    root = tk.Tk()
    app = RvcTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
