#!/usr/bin/env python3
"""Tkinter GUI for evaluating Asimov PPO models with saved action metadata.

The training backend saves ppo_asimov_push_final_metadata.json next to the
model. This GUI loads it automatically to map the PPO action vector to the
12 actuator actions. For old models without metadata, choose active actuator
indices manually in the GUI.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import tkinter as tk
from stable_baselines3 import PPO
from tkinter import filedialog, messagebox, ttk

from asimov_balance_env import ACTUATORS, AsimovBalanceEnv

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_XML = ROOT_DIR / "asimov0.xml"


class AsimovEvalGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Asimov PPO Evaluation")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.env: Optional[AsimovBalanceEnv] = None
        self.model: Optional[PPO] = None
        self.worker: Optional[threading.Thread] = None
        self.running = False
        self.stop_event = threading.Event()
        self.update_queue: list[tuple] = []
        self.queue_lock = threading.Lock()
        self.metadata: Optional[dict] = None

        self._build_config()
        self._build_telemetry()
        self._build_status()
        self.root.after(50, self._poll_updates)

    def _build_config(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Konfiguration", padding=8)
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        frame.columnconfigure(1, weight=1)

        rows = [
            ("PPO Model:", "model", ""),
            ("MuJoCo XML:", "xml", str(DEFAULT_XML)),
            ("Episodes:", "episodes", "4"),
            ("Push force [N]:", "force", "0"),
            ("Push start [s]:", "start", "2.0"),
            ("Push duration [s]:", "duration", "0.2"),
            ("Push direction:", "direction", "1,0,0"),
            ("Update every:", "every", "5"),
            ("Inactive action:", "fixed_action", "0.0"),
            ("Fallback indices:", "fallback_indices", ""),
        ]
        self.entries: dict[str, ttk.Entry] = {}
        for row, (label, key, value) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            entry = ttk.Entry(frame, width=80)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            self.entries[key] = entry

        ttk.Button(frame, text="Model wählen…", command=self._browse_model).grid(row=0, column=2, padx=4)
        ttk.Button(frame, text="XML wählen…", command=self._browse_xml).grid(row=1, column=2, padx=4)

        self.realtime = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Realtime MuJoCo", variable=self.realtime).grid(row=10, column=0, sticky="w", pady=5)
        self.metadata_status = tk.StringVar(value="Metadaten: Noch kein Modell ausgewählt")
        ttk.Label(frame, textvariable=self.metadata_status).grid(row=10, column=1, sticky="w", padx=6)

        buttons = ttk.Frame(frame)
        buttons.grid(row=10, column=2, sticky="e")
        self.start_button = ttk.Button(buttons, text="Start", command=self.start)
        self.start_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)

    def _build_telemetry(self) -> None:
        outer = ttk.LabelFrame(self.root, text="Telemetrie", padding=8)
        outer.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=4)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        sensors = ttk.LabelFrame(outer, text="SENSORDATEN", padding=6)
        joints = ttk.LabelFrame(outer, text="GELENKDATEN", padding=6)
        sensors.grid(row=0, column=0, sticky="nsew", padx=3)
        joints.grid(row=0, column=1, sticky="nsew", padx=3)

        self.sensor_labels = []
        for i in range(14):
            label = ttk.Label(sensors, text="", font=("Consolas", 9), anchor="w")
            label.grid(row=i, column=0, sticky="w")
            self.sensor_labels.append(label)

        ttk.Label(joints, text="joint                      actual      rel      vel   target", font=("Consolas", 9)).grid(row=0, column=0, sticky="w")
        self.joint_labels = []
        for i in range(len(ACTUATORS)):
            label = ttk.Label(joints, text="", font=("Consolas", 9), anchor="w")
            label.grid(row=i + 1, column=0, sticky="w")
            self.joint_labels.append(label)

    def _build_status(self) -> None:
        self.status = tk.StringVar(value="Bereit")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", padding=4).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=5
        )

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="PPO-Modell auswählen",
            filetypes=[("PPO model", "*.zip"), ("Alle Dateien", "*.*")],
        )
        if path:
            self.entries["model"].delete(0, tk.END)
            self.entries["model"].insert(0, path)
            self._load_metadata(Path(path))

    def _browse_xml(self) -> None:
        path = filedialog.askopenfilename(
            title="MuJoCo XML auswählen",
            filetypes=[("MuJoCo XML", "*.xml"), ("Alle Dateien", "*.*")],
        )
        if path:
            self.entries["xml"].delete(0, tk.END)
            self.entries["xml"].insert(0, path)

    def _load_metadata(self, model_path: Path) -> None:
        candidates = [
            model_path.with_name(model_path.stem + "_metadata.json"),
            model_path.parent / "ppo_asimov_push_final_metadata.json",
        ]
        self.metadata = None
        for candidate in candidates:
            if candidate.is_file():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    indices = np.asarray(data.get("active_indices"), dtype=np.int64)
                    fixed = np.asarray(data.get("fixed_action"), dtype=np.float32)
                    if indices.ndim != 1 or indices.size == 0:
                        raise ValueError("active_indices fehlen oder sind ungültig")
                    if fixed.shape != (len(ACTUATORS),):
                        raise ValueError("fixed_action muss 12 Werte enthalten")
                    self.metadata = data
                    self.entries["fallback_indices"].delete(0, tk.END)
                    self.entries["fallback_indices"].insert(0, ",".join(map(str, indices.tolist())))
                    self.entries["fixed_action"].delete(0, tk.END)
                    self.entries["fixed_action"].insert(0, ",".join(f"{value:g}" for value in fixed))
                    names = ", ".join(ACTUATORS[index] for index in indices)
                    self.metadata_status.set(f"Metadaten geladen: {candidate.name} | aktiv: {names}")
                    return
                except Exception as exc:
                    self.metadata_status.set(f"Metadaten konnten nicht gelesen werden: {exc}")
                    return
        self.metadata_status.set("Metadaten nicht gefunden: Fallback indices für altes Modell eingeben")

    @staticmethod
    def _parse_indices(text: str) -> np.ndarray:
        try:
            indices = np.asarray([int(x.strip()) for x in text.split(",") if x.strip()], dtype=np.int64)
        except ValueError as exc:
            raise ValueError("Fallback indices müssen Ganzzahlen mit Komma sein.") from exc
        if indices.size == 0:
            raise ValueError("Für Modelle ohne Metadaten müssen Fallback indices eingegeben werden.")
        if np.any(indices < 0) or np.any(indices >= len(ACTUATORS)):
            raise ValueError("Fallback indices müssen zwischen 0 und 11 liegen.")
        if np.unique(indices).size != indices.size:
            raise ValueError("Fallback indices dürfen keine Duplikate enthalten.")
        return indices

    @staticmethod
    def _parse_fixed_action(text: str) -> np.ndarray:
        try:
            if "," not in text:
                return np.full(len(ACTUATORS), float(text), dtype=np.float32)
            values = np.asarray([float(x.strip()) for x in text.split(",")], dtype=np.float32)
        except ValueError as exc:
            raise ValueError("Inactive action muss eine Zahl oder 12 Zahlen mit Komma sein.") from exc
        if values.shape != (len(ACTUATORS),):
            raise ValueError("Inactive action muss genau 12 Werte enthalten.")
        return values

    def _parse_config(self) -> dict:
        model_path = Path(self.entries["model"].get().strip()).expanduser()
        xml_path = Path(self.entries["xml"].get().strip()).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError("PPO-Modell nicht gefunden:\n" + str(model_path))
        if not xml_path.is_file():
            raise FileNotFoundError("MuJoCo XML nicht gefunden:\n" + str(xml_path))

        direction = np.asarray([float(x.strip()) for x in self.entries["direction"].get().split(",")], dtype=np.float64)
        if direction.shape != (3,) or np.linalg.norm(direction) < 1e-12:
            raise ValueError("Push direction muss z. B. -1,0,0 sein.")
        direction /= np.linalg.norm(direction)

        fixed_action = self._parse_fixed_action(self.entries["fixed_action"].get().strip())
        active_indices = self._parse_indices(self.entries["fallback_indices"].get().strip())
        return {
            "model_path": model_path.resolve(),
            "xml_path": xml_path.resolve(),
            "episodes": max(1, int(self.entries["episodes"].get())),
            "force_vec": direction * float(self.entries["force"].get()),
            "start": max(0.0, float(self.entries["start"].get())),
            "duration": max(0.0, float(self.entries["duration"].get())),
            "every": max(1, int(self.entries["every"].get())),
            "fixed_action": fixed_action,
            "active_indices": active_indices,
            "realtime": self.realtime.get(),
        }

    def start(self) -> None:
        if self.running:
            return
        try:
            config = self._parse_config()
        except Exception as exc:
            messagebox.showerror("Fehler", str(exc))
            return
        self.running = True
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status.set("Initialisiere Modell …")
        self.worker = threading.Thread(target=self._worker, args=(config,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.status.set("Stop angefordert …")

    def _worker(self, config: dict) -> None:
        try:
            self.model = PPO.load(str(config["model_path"]), device="auto")
            model_dim = int(self.model.action_space.shape[0])
            if model_dim < 1 or model_dim > len(ACTUATORS):
                raise ValueError(f"Nicht unterstützte PPO-Action-Dimension: ({model_dim},)")
            if model_dim != len(config["active_indices"]):
                raise ValueError(
                    f"Modell liefert {model_dim} Actions, aber die Metadaten/Fallback-Auswahl enthält "
                    f"{len(config['active_indices'])} aktive Gelenke."
                )

            self.env = AsimovBalanceEnv(
                model_path=config["xml_path"],
                render_mode="human" if config["realtime"] else None,
                control_dt=0.02,
                action_scale=float(self.metadata.get("action_scale", 0.20)) if self.metadata else 0.20,
                kp=np.asarray(self.metadata.get("kp_values", 60.0)) if self.metadata else 60.0,
                kv=np.asarray(self.metadata.get("kv_values", 6.0)) if self.metadata else 6.0,
                push_enabled=False,
            )
            start_steps = max(1, round(config["start"] / self.env.control_dt))
            duration_steps = max(1, round(config["duration"] / self.env.model.opt.timestep))

            for episode in range(config["episodes"]):
                if self.stop_event.is_set():
                    break
                self._run_episode(config, episode, start_steps, duration_steps)
            self._queue_status("Evaluation beendet")
        except Exception as exc:
            self._queue_error(str(exc))
        finally:
            if self.env is not None:
                self.env.close()
            self.env = None
            self._queue_done()

    def _run_episode(self, config: dict, episode: int, start_steps: int, duration_steps: int) -> None:
        assert self.env is not None and self.model is not None
        obs, _ = self.env.reset()
        remaining = 0
        step = 0
        done = False

        while not done and not self.stop_event.is_set():
            policy_action, _ = self.model.predict(obs, deterministic=True)
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
            full_action = config["fixed_action"].copy()
            full_action[config["active_indices"]] = np.clip(policy_action, -1.0, 1.0)
            self.env._apply_action(full_action)

            for _ in range(self.env.frame_skip):
                self.env.data.qfrc_applied[:] = 0.0
                self.env.data.xfrc_applied[:] = 0.0
                if remaining > 0:
                    self.env.data.xfrc_applied[self.env.pelvis_id, :3] = config["force_vec"]
                    remaining -= 1
                elif step >= start_steps and config["duration"] > 0.0:
                    remaining = duration_steps - 1
                    self.env.data.xfrc_applied[self.env.pelvis_id, :3] = config["force_vec"]
                mujoco.mj_step(self.env.model, self.env.data)

            obs = self.env._observation()
            _, tilt, height = self.env._upright_metrics()
            step += 1
            done = height < self.env.pelvis_min_height or tilt > self.env.max_tilt_rad or step >= self.env.max_episode_steps
            if step % config["every"] == 0:
                self._queue_update(full_action, config["active_indices"], episode + 1, step, remaining > 0)
            if config["realtime"]:
                self.env.render()
                time.sleep(self.env.control_dt)

        self._queue_status(f"Episode {episode + 1} beendet")

    def _queue_update(self, action, active_indices, episode, step, push_active) -> None:
        with self.queue_lock:
            self.update_queue.append(("update", action.copy(), active_indices.copy(), episode, step, push_active))

    def _queue_status(self, text: str) -> None:
        with self.queue_lock:
            self.update_queue.append(("status", text))

    def _queue_error(self, text: str) -> None:
        with self.queue_lock:
            self.update_queue.append(("error", text))

    def _queue_done(self) -> None:
        with self.queue_lock:
            self.update_queue.append(("done",))

    def _poll_updates(self) -> None:
        with self.queue_lock:
            events = self.update_queue[:]
            self.update_queue.clear()
        for event in events:
            if event[0] == "update":
                self._update_telemetry(*event[1:])
            elif event[0] == "status":
                self.status.set(event[1])
            elif event[0] == "error":
                messagebox.showerror("Fehler", event[1])
            elif event[0] == "done":
                self.running = False
                self.start_button.config(state="normal")
                self.stop_button.config(state="disabled")
        self.root.after(50, self._poll_updates)

    def _update_telemetry(self, action, active_indices, episode, step, push_active) -> None:
        env = self.env
        if env is None:
            return
        upright_cosine, tilt, height = env._upright_metrics()
        gravity = env.projected_gravity(env._sensor("imu_quat"))
        omega = env._sensor("imu_ang_vel")
        raw, binary = env._foot_contacts()
        drift = float(np.linalg.norm(env.data.xpos[env.pelvis_id, :2] - env.initial_pelvis_xy))
        lines = [
            f"tilt              : {tilt:8.4f} rad ({np.degrees(tilt):6.2f} deg)",
            f"upright cosine    : {upright_cosine:8.4f}",
            f"gravity pelvis    : [{gravity[0]:7.3f}, {gravity[1]:7.3f}, {gravity[2]:7.3f}]",
            f"omega x/y/z       : [{omega[0]:7.3f}, {omega[1]:7.3f}, {omega[2]:7.3f}] rad/s",
            f"pelvis height     : {height:8.4f} m",
            f"drift             : {drift:8.4f} m",
            f"left contact raw  : {raw[0]:8.3f}",
            f"right contact raw : {raw[1]:8.3f}",
            f"left contact bin  : {int(binary[0])}",
            f"right contact bin : {int(binary[1])}",
            f"push active       : {push_active}",
            f"active indices    : {active_indices.tolist()}",
            f"episode           : {episode}",
            f"control step      : {step}",
        ]
        for label, text in zip(self.sensor_labels, lines):
            label.config(text=text)

        q = env._joint_positions()
        rel = q - env.default_qpos
        vel = env._joint_velocities()
        targets = env.default_qpos + env.action_scale * action
        active_set = set(int(index) for index in active_indices)
        for i, name in enumerate(ACTUATORS):
            marker = "PPO" if i in active_set else "FIX"
            self.joint_labels[i].config(
                text=f"{marker:3s} {name:26s} {q[i]:8.3f} {rel[i]:8.3f} {vel[i]:8.3f} {targets[i]:8.3f}"
            )
        self.status.set(f"Episode {episode} | Step {step} | Push={push_active}")

    def close(self) -> None:
        self.stop_event.set()
        if self.env is not None:
            self.env.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    AsimovEvalGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
