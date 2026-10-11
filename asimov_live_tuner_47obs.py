#!/usr/bin/env python3
"""Asimov Dynamics Lab with original GUI and named 47-value PPO observation."""

import argparse
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np

DEFAULT = Path("./asimov0.xml")

ACT = [
    "left_hip_pitch_act", "left_hip_roll_act", "left_hip_yaw_act",
    "left_knee_act", "left_ankle_pitch_act", "left_ankle_roll_act",
    "right_hip_pitch_act", "right_hip_roll_act", "right_hip_yaw_act",
    "right_knee_act", "right_ankle_pitch_act", "right_ankle_roll_act",
]

SQUAT = {
    "left_hip_pitch_act": -.25, "left_knee_act": .35,
    "left_ankle_pitch_act": -.10, "right_hip_pitch_act": .25,
    "right_knee_act": -.35, "right_ankle_pitch_act": .10,
}

LEAN = {
    "left_hip_roll_act": .11, "right_hip_roll_act": .11,
    "left_ankle_roll_act": .09, "right_ankle_roll_act": .09,
}

SENSOR_NAMES = (
    "imu_ang_vel", "imu_lin_vel", "imu_quat", "root_angmom",
    "left_foot_contact", "right_foot_contact",
)


class Lab:
    def __init__(self, path):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

        self.aid = {n: self.m.actuator(n).id for n in ACT}
        self.qadr = {}
        self.dofadr = {}
        for name, actuator_id in self.aid.items():
            joint_id = int(self.m.actuator_trnid[actuator_id, 0])
            self.qadr[name] = int(self.m.jnt_qposadr[joint_id])
            self.dofadr[name] = int(self.m.jnt_dofadr[joint_id])

        self.pelvis = self.m.body("pelvis_link").id
        self.toes = [
            self.m.joint("left_toe_joint").id,
            self.m.joint("right_toe_joint").id,
        ]
        self.contact_geoms = np.flatnonzero(
            (self.m.geom_contype != 0)
            | (self.m.geom_conaffinity != 0)
        )
        self.q0 = self.d.qpos.copy()
        self.joint_q_default = np.asarray(
            [self.d.qpos[self.qadr[n]] for n in ACT],
            dtype=np.float32,
        )

        self.sensor_ids = {
            name: mujoco.mj_name2id(
                self.m,
                mujoco.mjtObj.mjOBJ_SENSOR,
                name,
            )
            for name in SENSOR_NAMES
        }
        missing = [n for n, sid in self.sensor_ids.items() if sid < 0]
        if missing:
            raise RuntimeError(
                "Missing XML sensors: " + ", ".join(missing)
            )
        self.sensor_adr = {
            name: (
                int(self.m.sensor_adr[sid]),
                int(self.m.sensor_dim[sid]),
            )
            for name, sid in self.sensor_ids.items()
        }

        self.previous_action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.mode = "MANUAL"
        self.paused = False
        self.selected = self.pelvis
        self.push = None
        self.push_steps = 0
        self.push_J = 0.0
        self.push_vec = np.zeros(3)
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Asimov Dynamics Lab")
        self.root.geometry("1100x940")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.x = {}
        self.j = {}
        self.out = {}
        self.build()

        self.v = mujoco.viewer.launch_passive(self.m, self.d)
        self.last_wall = time.perf_counter()
        self.acc = 0.0
        self.root.after(10, self.tick)
        self.root.mainloop()

    def scale(self, p, r, label, lo, hi, val, res):
        v = tk.DoubleVar(value=val)
        ttk.Label(p, text=label, width=24).grid(row=r, column=0, sticky="w")
        tk.Scale(
            p, variable=v, from_=lo, to=hi, resolution=res,
            orient="horizontal", length=410,
        ).grid(row=r, column=1, sticky="w")
        return v

    def build(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill="x")
        self.pause_text = tk.StringVar(value="Pause")
        ttk.Button(bar, textvariable=self.pause_text, command=self.pause).pack(side="left")
        ttk.Button(bar, text="Reset pose", command=self.reset_pose).pack(side="left", padx=4)
        ttk.Button(bar, text="RESET ALL", command=self.reset_all).pack(side="left")
        self.mode_text = ttk.Label(bar, text="Mode: MANUAL", foreground="#075cba")
        self.mode_text.pack(side="right")

        book = ttk.Notebook(self.root)
        book.pack(fill="both", expand=True, padx=6)
        J = ttk.Frame(book, padding=6)
        G = ttk.Frame(book, padding=6)
        P = ttk.Frame(book, padding=6)
        I = ttk.Frame(book, padding=6)
        S = ttk.Frame(book, padding=6)
        for frame, name in [
            (J, "Joint control"), (G, "Pose & gait"),
            (P, "Physics & ground"), (I, "Inspect & push"),
            (S, "Sensors / PPO"),
        ]:
            book.add(frame, text=name)

        ttk.Label(
            J,
            text="Move any slider to enter MANUAL mode. Readout: target / actual / effort",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        for r, n in enumerate(ACT, 1):
            a = self.aid[n]
            lo, hi = self.m.actuator_ctrlrange[a]
            ttk.Label(J, text=n[:-4], width=24).grid(row=r, column=0, sticky="w")
            v = tk.DoubleVar(value=0)
            tk.Scale(
                J, variable=v, from_=lo, to=hi, resolution=.005,
                orient="horizontal", length=410,
                command=lambda _, n=n: self.set_mode("MANUAL"),
            ).grid(row=r, column=1)
            o = ttk.Label(J, width=34, font=("TkFixedFont", 9))
            o.grid(row=r, column=2, sticky="w")
            self.j[n] = v
            self.out[n] = o

        buttons = ttk.Frame(G)
        buttons.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        for text, mode in [("Manual joints", "MANUAL"), ("Squat / lean", "POSE"), ("Open-loop walk", "WALK")]:
            ttk.Button(buttons, text=text, command=lambda m=mode: self.set_mode(m)).pack(side="left", padx=3)
        self.x["squat"] = self.scale(G, 1, "Squat depth", 0, 1, 0, .01)
        self.x["lean"] = self.scale(G, 2, "Lateral lean", -1, 1, 0, .01)
        self.x["hz"] = self.scale(G, 3, "Walk frequency (Hz)", .05, 2, .55, .01)
        self.x["amp"] = self.scale(G, 4, "Walk amplitude", 0, 1, .65, .01)
        self.gait_status = ttk.Label(G, font=("TkFixedFont", 10))
        self.gait_status.grid(row=5, column=0, columnspan=2, sticky="w", pady=10)

        self.x["kp"] = self.scale(P, 0, "Position gain kp", 0, 300, 60, 1)
        self.x["kv"] = self.scale(P, 1, "Velocity gain kv", 0, 40, 6, .1)
        self.x["slip"] = self.scale(P, 2, "Ground slippiness", 0, 1, .16, .01)
        self.slip_label = ttk.Label(P)
        self.slip_label.grid(row=3, column=0, columnspan=2, sticky="w")
        self.x["tk"] = self.scale(P, 4, "Toe stiffness", 0, 100, 4.5, .5)
        self.x["td"] = self.scale(P, 5, "Toe damping", 0, 10, .5, .05)
        self.x["dt"] = self.scale(P, 6, "Timestep (ms)", .2, 10, 2, .1)

        self.sel_title = ttk.Label(I, text="Selected: pelvis_link", font=("TkDefaultFont", 13, "bold"))
        self.sel_title.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(I, text="Double-click a visible body in the MuJoCo viewer to select it. Pushes target its center of mass.").grid(row=1, column=0, columnspan=3, sticky="w")
        self.spec = tk.Text(I, width=108, height=17, font=("TkFixedFont", 9), state="disabled")
        self.spec.grid(row=2, column=0, columnspan=3, pady=5)
        self.x["force"] = self.scale(I, 3, "Pulse force (N)", 0, 500, 100, 5)
        self.x["dur"] = self.scale(I, 4, "Pulse duration (s)", .02, 2, .20, .01)
        b = ttk.Frame(I)
        b.grid(row=6, column=0, columnspan=3, pady=6)
        for text, u in [("Forward +X", (1, 0, 0)), ("Back −X", (-1, 0, 0)), ("Left +Y", (0, 1, 0)), ("Right −Y", (0, -1, 0)), ("Up +Z", (0, 0, 1))]:
            ttk.Button(b, text=text, command=lambda u=u: self.start_push(u)).pack(side="left", padx=3)
        self.push_status = ttk.Label(I, font=("TkFixedFont", 9))
        self.push_status.grid(row=7, column=0, columnspan=3, sticky="w")

        ttk.Label(S, text="Named PPO observation (47 values) and raw XML sensor values").pack(anchor="w")
        self.sensor_text = tk.Text(S, width=120, height=48, font=("TkFixedFont", 9), state="disabled")
        self.sensor_text.pack(fill="both", expand=True, pady=6)

        self.status = ttk.Label(self.root, relief="sunken", padding=6)
        self.status.pack(fill="x", side="bottom")
        self.defaults = {k: v.get() for k, v in self.x.items()}
        self.show_specs(self.pelvis)

    def set_mode(self, mode):
        self.mode = mode
        self.mode_text.configure(text="Mode: " + mode)

    def pause(self):
        self.paused = not self.paused
        self.pause_text.set("Resume" if self.paused else "Pause")

    def reset_pose(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.q0
        self.d.ctrl[:] = 0
        self.d.qfrc_applied[:] = 0
        self.d.xfrc_applied[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.push = None
        self.push_steps = 0
        self.push_J = 0.0
        self.push_vec[:] = 0.0
        self.previous_action[:] = 0.0
        self.acc = 0.0

    def reset_all(self):
        self.reset_pose()
        self.set_mode("MANUAL")
        for v in self.j.values():
            v.set(0)
        for k, v in self.x.items():
            v.set(self.defaults[k])

    def generated_targets(self):
        if self.mode == "MANUAL":
            return {n: v.get() for n, v in self.j.items()}, 0.0
        targets = {n: 0.0 for n in ACT}
        phase = 0.0
        if self.mode == "POSE":
            for n, v in SQUAT.items():
                targets[n] += v * self.x["squat"].get()
            for n, v in LEAN.items():
                targets[n] += v * self.x["lean"].get()
        else:
            phase = 2 * math.pi * self.x["hz"].get() * self.d.time
            amplitude = self.x["amp"].get()
            s = math.sin(phase)
            left, right = max(0, s), max(0, -s)
            targets.update({"left_hip_pitch_act": -.38 * amplitude * s, "right_hip_pitch_act": .38 * amplitude * s, "left_knee_act": .70 * amplitude * left, "right_knee_act": -.70 * amplitude * right, "left_ankle_pitch_act": .26 * amplitude * left, "right_ankle_pitch_act": -.26 * amplitude * right})
        return targets, phase

    def start_push(self, u):
        self.push = (self.selected, np.array(u, float) * self.x["force"].get(), self.x["dur"].get())

    def apply(self):
        mu = .03 * (50 ** (1 - self.x["slip"].get()))
        self.m.geom_friction[self.contact_geoms] = (mu, .005, .0001)
        kp, kv = self.x["kp"].get(), self.x["kv"].get()
        self.m.actuator_gainprm[:, 0] = kp
        self.m.actuator_biasprm[:, 1] = -kp
        self.m.actuator_biasprm[:, 2] = -kv
        self.m.opt.timestep = self.x["dt"].get() / 1000
        self.m.jnt_stiffness[self.toes] = self.x["tk"].get()
        for j in self.toes:
            self.m.dof_damping[int(self.m.jnt_dofadr[j])] = self.x["td"].get()
        targets, phase = self.generated_targets()
        action = np.zeros(12, dtype=np.float32)
        for index, n in enumerate(ACT):
            a = self.aid[n]
            self.d.ctrl[a] = np.clip(targets[n], *self.m.actuator_ctrlrange[a])
            action[index] = self.d.ctrl[a]
        if self.push:
            self.push_body, self.push_vec, dur = self.push
            self.push_steps = max(1, math.ceil(dur / self.m.opt.timestep))
            self.push_J = 0.0
            self.push = None
        self.d.qfrc_applied[:] = 0
        if self.push_steps > 0:
            mujoco.mj_applyFT(self.m, self.d, self.push_vec, np.zeros(3), self.d.xipos[self.push_body], self.push_body, self.d.qfrc_applied)
            self.push_J += np.linalg.norm(self.push_vec) * self.m.opt.timestep
            self.push_steps -= 1
        mujoco.mj_step(self.m, self.d)
        self.previous_action[:] = action
        return mu, phase, targets

    def sensor_values(self):
        def read(name):
            adr, dim = self.sensor_adr[name]
            return self.d.sensordata[adr:adr + dim].copy()
        return {name: read(name) for name in SENSOR_NAMES}

    @staticmethod
    def projected_gravity(quat):
        w, x, y, z = quat
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float32)
        return rotation.T @ np.array([0, 0, -1], dtype=np.float32)

    def build_observation(self):
        s = self.sensor_values()
        gravity = self.projected_gravity(s["imu_quat"])
        q = np.asarray([self.d.qpos[self.qadr[n]] for n in ACT], dtype=np.float32) - self.joint_q_default
        dq = np.asarray([self.d.qvel[self.dofadr[n]] for n in ACT], dtype=np.float32)
        contacts = np.array([
            float(s["left_foot_contact"][0] > 5.0),
            float(s["right_foot_contact"][0] > 5.0),
        ], dtype=np.float32)
        obs = np.concatenate([
            gravity, s["imu_ang_vel"].astype(np.float32), q, dq,
            contacts, self.previous_action, self.command,
        ]).astype(np.float32)
        if obs.shape != (47,):
            raise RuntimeError(f"Expected 47 observation values, got {obs.shape}")
        return obs

    def update_sensor_page(self, obs):
        s = self.sensor_values()
        names = [
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
            "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
        ]
        names += [f"{n}_q_rel" for n in ACT]
        names += [f"{n}_qvel" for n in ACT]
        names += ["left_foot_contact", "right_foot_contact"]
        names += [f"{n}_previous_action" for n in ACT]
        names += ["command_vx", "command_vy", "command_yaw_rate"]
        lines = [
            "PPO policy observation: 47 values",
            "================================",
            "",
            "Raw XML sensor values",
            "---------------------",
        ]
        for sensor_name in SENSOR_NAMES:
            lines.append(f"{sensor_name:22s}: {np.array2string(s[sensor_name], precision=6)}")
        lines += ["", "Named PPO observation", "---------------------"]
        lines += [f"obs[{i:02d}] {name:42s} = {value:+.6f}" for i, (name, value) in enumerate(zip(names, obs))]
        self.sensor_text.configure(state="normal")
        self.sensor_text.delete("1.0", "end")
        self.sensor_text.insert("1.0", "\n".join(lines))
        self.sensor_text.configure(state="disabled")

    def tick(self):
        if self.closed or not self.v.is_running():
            self.close()
            return
        now = time.perf_counter()
        self.acc += min(now - self.last_wall, .05)
        self.last_wall = now
        mu, phase, targets = .8, 0.0, {}
        if not self.paused:
            steps = 0
            while self.acc >= self.m.opt.timestep and steps < 100:
                with self.v.lock():
                    mu, phase, targets = self.apply()
                    obs = self.build_observation()
                self.acc -= self.m.opt.timestep
                steps += 1
            if steps:
                self.update_sensor_page(obs)
        self.v.sync()
        self.update_ui(mu, phase, targets)
        self.root.after(10, self.tick)

    def update_ui(self, mu, phase, targets):
        for n in ACT:
            self.out[n].configure(text=f"target={self.d.ctrl[self.aid[n]]:+.3f} actual={self.d.qpos[self.qadr[n]]:+.3f} effort={self.d.actuator_force[self.aid[n]]:+.2f}")
        self.slip_label.configure(text=f"Current effective coefficient μ={mu:.3f}")
        active = self.push_steps > 0
        self.push_status.configure(text=f"{'PUSH ACTIVE' if active else 'idle':11s} command={np.linalg.norm(self.push_vec) if active else 0:6.1f} N delivered impulse={self.push_J:6.2f} N·s steps left={self.push_steps:4d}")
        x, y, z = self.d.xpos[self.pelvis]
        self.status.configure(text=f"{'PAUSED' if self.paused else 'RUNNING'} t={self.d.time:7.2f}s pelvis=({x:+.3f},{y:+.3f},{z:.3f}) m contacts={self.d.ncon}")

    def body_info(self, b):
        m = self.m
        lines = [f"body id : {b}", f"name : {m.body(b).name}", f"parent : {m.body(int(m.body_parentid[b])).name}", f"mass : {m.body_mass[b]:.6g} kg", f"inertia : {m.body_inertia[b].tolist()} kg·m²", f"local COM : {m.body_ipos[b].tolist()} m"]
        start = int(m.body_jntadr[b])
        end = start + int(m.body_jntnum[b])
        for j in range(start, end):
            lines += [f"joint : {m.joint(j).name}", f"axis : {m.jnt_axis[j].tolist()}", f"range : {m.jnt_range[j].tolist()} rad", f"stiffness : {m.jnt_stiffness[j]:g}"]
        return "\n".join(lines)

    def show_specs(self, b):
        self.sel_title.configure(text="Selected: " + self.m.body(b).name)
        self.spec.configure(state="normal")
        self.spec.delete("1.0", "end")
        self.spec.insert("1.0", self.body_info(b))
        self.spec.configure(state="disabled")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.v.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT)
    args = parser.parse_args()
    Lab(args.model.resolve())
