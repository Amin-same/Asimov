#!/usr/bin/env python3
"""Two-page Tkinter GUI for configurable Asimov PPO training and resume."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ACTUATORS = (
    "left_hip_pitch_act", "left_hip_roll_act", "left_hip_yaw_act",
    "left_knee_act", "left_ankle_pitch_act", "left_ankle_roll_act",
    "right_hip_pitch_act", "right_hip_roll_act", "right_hip_yaw_act",
    "right_knee_act", "right_ankle_pitch_act", "right_ankle_roll_act",
)

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT_DIR / "train_asimov_ppo_push.py"


class TrainingGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Asimov balance PPO Training")
        self.root.geometry("1080x780")
        self.root.minsize(930, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str | None] = queue.Queue()
        self._create_variables()
        self._build_ui()
        self.root.after(100, self._poll_output)

    def _create_variables(self) -> None:
        self.values = {
            "model_xml": tk.StringVar(value=str(ROOT_DIR / "asimov0.xml")),
            "run_name": tk.StringVar(value="asimov_push_selectable"),
            "resume_model": tk.StringVar(value=""),
            "timesteps": tk.StringVar(value="300000"),
            "n_envs": tk.StringVar(value="16"),
            "seed": tk.StringVar(value="42"),
            "min_force": tk.StringVar(value="4.0"),
            "max_force": tk.StringVar(value="5.0"),
            "push_prob": tk.StringVar(value="0.7"),
            "push_start_min": tk.StringVar(value="1.0"),
            "push_start_max": tk.StringVar(value="3.0"),
            "push_duration_min": tk.StringVar(value="0.1"),
            "push_duration_max": tk.StringVar(value="0.5"),
            "action_scale": tk.StringVar(value="0.2"),
            "max_episode_seconds": tk.StringVar(value="5"),
            "contact_threshold": tk.StringVar(value="20.0"),
            "pelvis_min_height": tk.StringVar(value="0.35"),
            "max_tilt_rad": tk.StringVar(value="0.7"),
            "learning_rate": tk.StringVar(value="0.0003"),
            "n_steps": tk.StringVar(value="1024"),
            "batch_size": tk.StringVar(value="64"),
            "n_epochs": tk.StringVar(value="10"),
            "gamma": tk.StringVar(value="0.99"),
            "gae_lambda": tk.StringVar(value="0.95"),
            "clip_range": tk.StringVar(value="0.2"),
            "ent_coef": tk.StringVar(value="0.05"),
        }
        self.mode = tk.StringVar(value="new")
        self.visualize = tk.BooleanVar(value=False)
        default_active = {
            "left_ankle_pitch_act", "left_ankle_roll_act",
            "right_ankle_pitch_act", "right_ankle_roll_act",
        }
        self.active = {name: tk.BooleanVar(value=name in default_active) for name in ACTUATORS}
        self.kp = {name: tk.StringVar(value="60.0") for name in ACTUATORS}
        self.kv = {name: tk.StringVar(value="6.0") for name in ACTUATORS}
        self.status = tk.StringVar(value="Bereit")

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        page1 = ttk.Frame(notebook, padding=10)
        page2 = ttk.Frame(notebook, padding=10)
        notebook.add(page1, text="Training-Parameter")
        notebook.add(page2, text="Gelenke und PD-Regler")
        self._build_parameters_page(page1)
        self._build_joints_page(page2)
        bottom = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status).pack(side="left")
        self.stop_button = ttk.Button(bottom, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="right", padx=(6, 0))
        self.start_button = ttk.Button(bottom, text="Training starten", command=self.start)
        self.start_button.pack(side="right")

    def _build_parameters_page(self, page: ttk.Frame) -> None:
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=1)

        mode_frame = ttk.LabelFrame(page, text="Trainingsmodus", padding=8)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(mode_frame, text="Neues Training", variable=self.mode, value="new", command=self._update_mode).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Radiobutton(mode_frame, text="Bestehendes Modell fortsetzen", variable=self.mode, value="resume", command=self._update_mode).grid(row=0, column=1, sticky="w")
        ttk.Label(mode_frame, text="Fortsetzen ist nur mit identischer Auswahl und Reihenfolge der aktiven Gelenke möglich.").grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.resume_entry = self._add_entry(mode_frame, 2, "PPO-Modell (.zip)", "resume_model", browse="model")

        left = ttk.LabelFrame(page, text="Training und Push", padding=8)
        left.grid(row=1, column=0, sticky="new", padx=(0, 5))
        right = ttk.LabelFrame(page, text="Environment und PPO", padding=8)
        right.grid(row=1, column=1, sticky="new", padx=5)
        self._add_entry(left, 0, "MuJoCo XML", "model_xml", browse="xml")
        self._add_entry(left, 1, "Run-Name", "run_name")
        self._add_entry(left, 2, "Zusätzliche Timesteps", "timesteps")
        self._add_entry(left, 3, "Parallel Envs", "n_envs")
        self._add_entry(left, 4, "Seed", "seed")
        self._add_entry(left, 5, "Push force min [N]", "min_force")
        self._add_entry(left, 6, "Push force max [N]", "max_force")
        self._add_entry(left, 7, "Push probability", "push_prob")
        self._add_entry(left, 8, "Push start min [s]", "push_start_min")
        self._add_entry(left, 9, "Push start max [s]", "push_start_max")
        self._add_entry(left, 10, "Push duration min [s]", "push_duration_min")
        self._add_entry(left, 11, "Push duration max [s]", "push_duration_max")
        self._add_entry(right, 0, "Action scale", "action_scale")
        self._add_entry(right, 1, "Episode max [s]", "max_episode_seconds")
        self._add_entry(right, 2, "Contact threshold", "contact_threshold")
        self._add_entry(right, 3, "Pelvis min height [m]", "pelvis_min_height")
        self._add_entry(right, 4, "Max tilt [rad]", "max_tilt_rad")
        self._add_entry(right, 5, "Learning rate", "learning_rate")
        self._add_entry(right, 6, "n_steps", "n_steps")
        self._add_entry(right, 7, "Batch size", "batch_size")
        self._add_entry(right, 8, "n_epochs", "n_epochs")
        self._add_entry(right, 9, "Gamma", "gamma")
        self._add_entry(right, 10, "GAE lambda", "gae_lambda")
        self._add_entry(right, 11, "Clip range", "clip_range")
        self._add_entry(right, 12, "Entropy coefficient", "ent_coef")
        ttk.Checkbutton(page, text="MuJoCo visualisieren (setzt Parallel Envs = 1)", variable=self.visualize).grid(row=2, column=0, columnspan=2, sticky="w", pady=10)

        output_frame = ttk.LabelFrame(page, text="Trainingsausgabe", padding=6)
        output_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        page.rowconfigure(3, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.output = tk.Text(output_frame, state="disabled", wrap="word", font=("Consolas", 9))
        self.output.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)
        self._update_mode()

    def _build_joints_page(self, page: ttk.Frame) -> None:
        ttk.Label(page, text="Häkchen = PPO lernt dieses Gelenk. Ohne Häkchen bleibt die Action dieses Gelenks auf 0.0. Kp und Kv gelten immer für den jeweiligen PD-Regler.", wraplength=960).pack(anchor="w", pady=(0, 10))
        table = ttk.LabelFrame(page, text="Aktuatorauswahl", padding=8)
        table.pack(fill="x", anchor="n")
        for col, text in enumerate(("Lernen", "Index", "Gelenkname", "Kp", "Kv")):
            ttk.Label(table, text=text).grid(row=0, column=col, sticky="w", padx=6, pady=(0, 6))
        for row, name in enumerate(ACTUATORS, start=1):
            ttk.Checkbutton(table, variable=self.active[name]).grid(row=row, column=0, sticky="w", padx=6, pady=2)
            ttk.Label(table, text=str(row - 1)).grid(row=row, column=1, sticky="w", padx=6)
            ttk.Label(table, text=name, width=28).grid(row=row, column=2, sticky="w", padx=6)
            ttk.Entry(table, textvariable=self.kp[name], width=12).grid(row=row, column=3, padx=6)
            ttk.Entry(table, textvariable=self.kv[name], width=12).grid(row=row, column=4, padx=6)
        buttons = ttk.Frame(page)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="Alle auswählen", command=lambda: self._set_all(True)).pack(side="left", padx=(0, 5))
        ttk.Button(buttons, text="Alle abwählen", command=lambda: self._set_all(False)).pack(side="left", padx=5)
        ttk.Button(buttons, text="Nur Knie + Knöchel", command=self._select_knees_ankles).pack(side="left", padx=5)
        ttk.Button(buttons, text="Kp/Kv Standard", command=self._reset_gains).pack(side="left", padx=5)

    def _add_entry(self, parent, row: int, label: str, key: str, browse: str | None = None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        entry = ttk.Entry(parent, textvariable=self.values[key], width=35)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        if browse:
            ttk.Button(parent, text="Browse…", command=lambda: self._browse(browse)).grid(row=row, column=2, padx=4)
        return entry

    def _browse(self, kind: str) -> None:
        if kind == "xml":
            path = filedialog.askopenfilename(filetypes=[("MuJoCo XML", "*.xml"), ("All files", "*.*")])
            if path:
                self.values["model_xml"].set(path)
        else:
            path = filedialog.askopenfilename(filetypes=[("PPO model", "*.zip"), ("All files", "*.*")])
            if path:
                self.values["resume_model"].set(path)
                parent = Path(path).resolve().parent
                if parent.name == "models" and parent.parent.name:
                    self.values["run_name"].set(parent.parent.name)

    def _update_mode(self) -> None:
        enabled = self.mode.get() == "resume"
        self.resume_entry.configure(state="normal" if enabled else "disabled")

    def _set_all(self, value: bool) -> None:
        for variable in self.active.values():
            variable.set(value)

    def _select_knees_ankles(self) -> None:
        selected = {"left_knee_act", "left_ankle_pitch_act", "left_ankle_roll_act", "right_knee_act", "right_ankle_pitch_act", "right_ankle_roll_act"}
        for name in ACTUATORS:
            self.active[name].set(name in selected)

    def _reset_gains(self) -> None:
        for name in ACTUATORS:
            self.kp[name].set("60.0")
            self.kv[name].set("6.0")

    def _command(self) -> list[str]:
        if not TRAIN_SCRIPT.exists():
            raise ValueError(f"Trainingsskript nicht gefunden: {TRAIN_SCRIPT}")
        indices = [str(i) for i, name in enumerate(ACTUATORS) if self.active[name].get()]
        if not indices:
            raise ValueError("Mindestens ein Gelenk muss ausgewählt werden.")
        kp_values = [self.kp[name].get().strip() for name in ACTUATORS]
        kv_values = [self.kv[name].get().strip() for name in ACTUATORS]
        command = [sys.executable, str(TRAIN_SCRIPT)]
        mapping = {
            "--model-xml": "model_xml", "--run-name": "run_name", "--timesteps": "timesteps", "--n-envs": "n_envs", "--seed": "seed",
            "--min-push-force": "min_force", "--max-push-force": "max_force", "--push-prob": "push_prob", "--push-start-min": "push_start_min",
            "--push-start-max": "push_start_max", "--push-duration-min": "push_duration_min", "--push-duration-max": "push_duration_max",
            "--action-scale": "action_scale", "--max-episode-seconds": "max_episode_seconds", "--contact-threshold": "contact_threshold",
            "--pelvis-min-height": "pelvis_min_height", "--max-tilt-rad": "max_tilt_rad", "--learning-rate": "learning_rate", "--n-steps": "n_steps",
            "--batch-size": "batch_size", "--n-epochs": "n_epochs", "--gamma": "gamma", "--gae-lambda": "gae_lambda", "--clip-range": "clip_range", "--ent-coef": "ent_coef",
        }
        for option, key in mapping.items():
            value = self.values[key].get().strip()
            if not value:
                raise ValueError(f"Feld darf nicht leer sein: {key}")
            command.extend([option, value])
        command.extend(["--active-indices", ",".join(indices)])
        command.extend(["--kp-values", ",".join(kp_values)])
        command.extend(["--kv-values", ",".join(kv_values)])
        if self.mode.get() == "resume":
            resume_path = Path(self.values["resume_model"].get().strip())
            if not resume_path.is_file():
                raise ValueError("Bitte ein vorhandenes PPO-Modell (.zip) zum Fortsetzen auswählen.")
            command.extend(["--resume", str(resume_path.resolve())])
        if self.visualize.get():
            command.append("--visualize")
        return command

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        try:
            command = self._command()
        except Exception as exc:
            messagebox.showerror("Ungültige Konfiguration", str(exc))
            return
        self._write("\n" + "=" * 78 + "\n")
        self._write("Starte Training:\n" + subprocess.list2cmdline(command) + "\n")
        self._write("=" * 78 + "\n")
        try:
            self.process = subprocess.Popen(command, cwd=str(ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status.set("Training läuft …")
            threading.Thread(target=self._read_process, daemon=True).start()
        except Exception as exc:
            messagebox.showerror("Startfehler", str(exc))

    def _read_process(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.output_queue.put(line)
        self.output_queue.put(None)

    def _poll_output(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item is None:
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status.set("Bereit")
                    self.process = None
                else:
                    self._write(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_output)

    def _write(self, text: str) -> None:
        self.output.config(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.config(state="disabled")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.status.set("Stop angefordert …")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
