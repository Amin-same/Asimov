#!/usr/bin/env python3
"""Gymnasium environment for Asimov PPO walking and push recovery."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import subprocess
import sys
import csv
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time

DEFAULT_MODEL = Path("./asimov0.xml")

ACTUATORS = (
    "left_hip_pitch_act",
    "left_hip_roll_act",
    "left_hip_yaw_act",
    "left_knee_act",
    "left_ankle_pitch_act",
    "left_ankle_roll_act",
    "right_hip_pitch_act",
    "right_hip_roll_act",
    "right_hip_yaw_act",
    "right_knee_act",
    "right_ankle_pitch_act",
    "right_ankle_roll_act",
)

REQUIRED_SENSORS = (
    "imu_ang_vel",
    "imu_quat",
    "left_foot_contact",
    "right_foot_contact",
)


class AsimovWalkEnv(gym.Env):
    """Standing-balance environment with optional random horizontal pushes."""

    metadata = {"render_modes": ["human", None], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        render_mode: str | None = None,
        control_dt: float = 0.02,
        action_scale: float = 0.20,
        max_episode_seconds: float = 13.0,
        contact_threshold: float = 5.0,
        pelvis_min_height: float = 0.35,
        max_tilt_rad: float = 0.70,
        kp: float | np.ndarray = 60.0,
        kv: float | np.ndarray = 6.0,
        joint_acceleration_penalty_weight: float = 0.02,
        push_enabled: bool = False,
        push_probability: float = 0.70,
        min_push_force: float = 4.0,
        max_push_force: float = 6.0,
        push_start_min: float = 4.0,
        push_start_max: float = 6.0,
        push_duration_min: float = 0.1,
        push_duration_max: float = 0.3,
        recovery_hold_seconds: float = 1.0,
    ):
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if not 0.0 <= push_probability <= 1.0:
            raise ValueError("push_probability must be between 0.0 and 1.0")
        if min_push_force < 0.0 or max_push_force < min_push_force:
            raise ValueError("Push-force limits are invalid")
        if push_start_min < 0.0 or push_start_max < push_start_min:
            raise ValueError("Push start-time limits are invalid")
        if push_duration_min <= 0.0 or push_duration_max < push_duration_min:
            raise ValueError("Push duration limits are invalid")

        self.episode_counter = 0
        self.model_path = Path(model_path).resolve()
        self.render_mode = render_mode
        self.control_dt = float(control_dt)
        self.action_scale = float(action_scale)
        self.max_episode_seconds = float(max_episode_seconds)
        self.contact_threshold = float(contact_threshold)
        self.pelvis_min_height = float(pelvis_min_height)
        self.max_tilt_rad = float(max_tilt_rad)
        self.kp = self._gain_array(kp, "kp")
        self.kv = self._gain_array(kv, "kv")
        self.joint_acceleration_penalty_weight = float(joint_acceleration_penalty_weight)

        if self.joint_acceleration_penalty_weight < 0.0:
            raise ValueError("joint_acceleration_penalty_weight must be non-negative.")
        self.push_enabled = bool(push_enabled)
        self.push_probability = float(push_probability)
        self.min_push_force = float(min_push_force)
        self.max_push_force = float(max_push_force)
        self.push_start_min = float(push_start_min)
        self.push_start_max = float(push_start_max)
        self.push_duration_min = float(push_duration_min)
        self.push_duration_max = float(push_duration_max)
        self.recovery_hold_seconds = float(recovery_hold_seconds)

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.frame_skip = max(1, round(self.control_dt / self.model.opt.timestep))
        self.control_dt = self.frame_skip * self.model.opt.timestep

        self.actuator_ids = np.asarray([self._named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATORS],dtype=np.int32,)
        self.joint_ids = self.model.actuator_trnid[self.actuator_ids, 0].astype(np.int32)
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.control_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(np.float32)
        self.control_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(np.float32)
        if np.any(self.control_low >= self.control_high):
            raise ValueError(
                "Every position actuator needs ctrllimited='true' and a valid ctrlrange. "
                "Check the 12 actuator definitions in asimov0.xml.")
        self.pelvis_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self.left_ankle_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_ankle_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.stance_foot = 0
        self.swing_foot = 1
        # Ein-Bein-Balance Training
        self.dx_swing_stance = 0.0
        self.previous_swing_height = 0.0
        self.height_progress_delta = 0.0
        self.step_height_target = 0.05  # 30 cm anheben
        self.step_hold_time = 0.5  # 0.5 Sekunden halten
        self.step_hold_steps = int(self.step_hold_time / self.control_dt)  # Schritte für 0.5s
        self.swing_start_pos = np.zeros(3, dtype=np.float32)
        self.stance_start_pos = np.zeros(3, dtype=np.float32)
        self.step_phase = 0
        self.step_success = False
        self.step_landed = False
        self.step_hold_counter = 0
        # CoM-Geschwindigkeit für Richtungsbelohnung
        self.previous_com_x_rel = 0.0
        self.previous_com_y_rel = 0.0
        self.com_velocity_x = 0.0
        self.com_velocity_y = 0.0
        # Phase 0: CoM seitlich auf Standfuß verlagern (Y-Richtung)
        self.com_x_target_phase0 = 0.05  # Leicht vorne, aber nicht wichtig
        self.com_y_target_phase0 = 0.0  # Wird dynamisch gesetzt
        # Phase 1: CoM diagonal bewegen (1/3 des Weges)
        self.com_x_target_phase1 = 0.1  # Ungefähr 1/3 von 0.20
        self.com_y_target_phase1 = 0.0
        # Phase 2: CoM am Zielpunkt (relativ zu Fußmitte)
        self.com_x_target_phase2 = 0.05
        self.com_y_target_phase2 = 0.0
        # Phase 3: CoM am Zielpunkt (relativ zu Fußmitte)
        self.com_x_target_phase3 = 0.05
        self.com_y_target_phase3 = 0.2
        # Toleranzen
        self.com_phase0_tolerance = 0.07
        self.com_phase1_tolerance = 0.08
        self.com_phase2_tolerance = 0.04
        self.com_phase3_tolerance = 0.04
        # Haltezeiten
        self.phase0_hold_steps = 5
        self.phase1_hold_steps = 3
        self.phase2_min_steps = int(0.5 / self.control_dt)  # 0.5 Sekunden = 25 Schritte bei 50Hz
        self.phase3_hold_steps = 10
        self.phase0_counter = 0
        self.phase1_counter = 0
        self.phase2_counter = 0
        self.sensor_slices: dict[str, slice] = {}
        for name in REQUIRED_SENSORS:
            sensor_id = self._named_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
            address = int(self.model.sensor_adr[sensor_id])
            dimension = int(self.model.sensor_dim[sensor_id])
            self.sensor_slices[name] = slice(address, address + dimension)

        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(len(ACTUATORS),),dtype=np.float32,)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(58,),dtype=np.float32,)

        self.default_qpos = np.zeros(len(ACTUATORS), dtype=np.float32)
        self.previous_action = np.zeros(len(ACTUATORS), dtype=np.float32)
        self.previous_joint_velocity = np.zeros(len(ACTUATORS),dtype=np.float32,)
        self.command = np.zeros(3, dtype=np.float32)
        self.initial_pelvis_xy = np.zeros(2, dtype=np.float32)
        self.initial_pelvis_height = 0.0
        self.elapsed_steps = 0
        self.max_episode_steps = max(1, round(self.max_episode_seconds / self.control_dt))
        self.viewer = None
        self.simulation_paused = False
        # Unter macOS kein zusätzliches Matplotlib-Fenster im mjpython-Prozess.
        self.reward_plotter = None
        if self.render_mode == "human" and sys.platform != "darwin":
            self.reward_plotter = RewardPlotter(max_points=600)
        self.push_scheduled = False
        self.push_started = False
        self.push_start_step = 0
        self.push_duration_steps = 0
        self.push_steps_remaining = 0
        self.push_force = np.zeros(3, dtype=np.float64)
        self.push_impulse = 0.0
        self.recovery_steps = 0
        self.recovered = False
        self.previous_tilt = 0.0
        self.previous_x_deviation = 0.0
        self.previous_y_deviation = 0.0
        self.previous_dx_deviation = 0.0
        self.previous_dy_deviation = 0.0
        self._set_pd_gains()
        # CSV Logging initialisieren
        
        self.reward_csv = open(
        "reward_debug.csv",
        "w",
        newline="",
        encoding="utf-8"
        )
        
        self.reward_writer = None
        self.plot_process = None
        plot_script = Path(__file__).parent / "reward_live_plot.py"
        if self.render_mode == "human":
            self.plot_process = subprocess.Popen(
            ["/Users/asn/miniforge3/envs/asimenv/bin/python",
            str(plot_script)])
            print("Plot PID:", self.plot_process.pid)

    @staticmethod
    def _gain_array(value: float | np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(len(ACTUATORS), float(array), dtype=np.float64)
        if array.shape != (len(ACTUATORS),):
            raise ValueError(f"{name} must be a scalar or contain exactly {len(ACTUATORS)} values; "f"got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"All {name} values must be finite.")
        if np.any(array < 0.0):
            raise ValueError(f"All {name} values must be non-negative.")
        return array

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Required MuJoCo object not found: {name}")
        return object_id

    def _set_pd_gains(self) -> None:
        self.model.actuator_gainprm[self.actuator_ids, 0] = self.kp
        self.model.actuator_biasprm[self.actuator_ids, 1] = -self.kp
        self.model.actuator_biasprm[self.actuator_ids, 2] = -self.kv

    def _sensor(self, name: str) -> np.ndarray:
        return self.data.sensordata[self.sensor_slices[name]].copy()

    @staticmethod
    def projected_gravity(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(quat, dtype=np.float64)
        rotation_world_from_body = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=np.float64,)
        return (rotation_world_from_body.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

    def _joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addresses].astype(np.float32).copy()

    def _joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.dof_addresses].astype(np.float32).copy()

    def _foot_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        raw = np.array(
            [
                float(self._sensor("left_foot_contact")[0]),
                float(self._sensor("right_foot_contact")[0]),
            ], dtype=np.float32,)
        return raw, (raw > self.contact_threshold).astype(np.float32)

    def _observation(self) -> np.ndarray:
        gravity = self.projected_gravity(self._sensor("imu_quat"))
        angular_velocity = self._sensor("imu_ang_vel").astype(np.float32)
        q_relative = self._joint_positions() - self.default_qpos
        q_velocity = self._joint_velocities()
        _, contacts = self._foot_contacts()
        left_foot_pos = (self._foot_position(0) - self.data.xpos[self.pelvis_id]).astype(np.float32)
        right_foot_pos = (self._foot_position(1) - self.data.xpos[self.pelvis_id]).astype(np.float32)
        swing_foot_pos = (self._foot_position(self.swing_foot) - self.swing_start_pos).astype(np.float32)
        step_features = np.concatenate([left_foot_pos, right_foot_pos, swing_foot_pos, np.array([self.step_phase / 4.0, float(self.stance_foot), ], dtype=np.float32,), ])
        observation = np.concatenate([gravity, angular_velocity, q_relative, q_velocity, contacts, self.previous_action, self.command, step_features, ]).astype(np.float32)
        if observation.shape != (58,):
            raise RuntimeError(f"Observation must have shape (58,), got {observation.shape}")
        return observation

    def _foot_position(self, foot: int) -> np.ndarray:
        body_id = (self.left_ankle_id if foot == 0 else self.right_ankle_id)
        return self.data.xpos[body_id].copy()

    def _foot_contact_value(self, foot: int) -> float:
        sensor_name = ("left_foot_contact" if foot == 0 else "right_foot_contact")
        return float(self._sensor(sensor_name)[0])

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        target = self.default_qpos + self.action_scale * action
        self.data.ctrl[self.actuator_ids] = np.clip(target, self.control_low, self.control_high)
        return action

    def _upright_metrics(self) -> tuple[float, float, float]:
        gravity = self.projected_gravity(self._sensor("imu_quat"))
        upright_cosine = float(np.clip(-gravity[2], -1.0, 1.0))
        tilt = float(np.arccos(upright_cosine))
        pelvis_height = float(self.data.xpos[self.pelvis_id, 2])
        return upright_cosine, tilt, pelvis_height

    def _get_com_position(self) -> np.ndarray:
        """
        Gibt die Position des Gesamtschwerpunkts (Center of Mass) zurück.
        Rückgabe: 3D-Vektor [x, y, z] in Weltkoordinaten.
        """
        return self.data.subtree_com[self.pelvis_id].copy()

    def _has_fallen(self) -> bool:
        _, tilt, pelvis_height = self._upright_metrics()
        return tilt > self.max_tilt_rad

    def _schedule_push(self) -> None:
        self.push_scheduled = self.push_enabled and bool(self.np_random.random() < self.push_probability)
        self.push_started = False
        self.push_steps_remaining = 0.0
        self.push_impulse = 0.0
        self.recovery_steps = 0.0
        self.recovered = False
        self.push_force[:] = 0.0
        if not self.push_scheduled:
            return
        start_seconds = float(self.np_random.uniform(self.push_start_min, self.push_start_max))
        duration_seconds = float(self.np_random.uniform(self.push_duration_min, self.push_duration_max))
        magnitude = float(self.np_random.uniform(self.min_push_force, self.max_push_force))
        direction = self.np_random.uniform(0.0, 2.0 * np.pi)
        #if np.pi/2 < direction < 3*np.pi/2:
            #magnitude = magnitude/3
        angle = float(direction)
        self.push_start_step = max(1, round(start_seconds / self.control_dt))
        self.push_duration_steps = max(1, round(duration_seconds / self.model.opt.timestep))
        self.push_force[:] = (magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0)
    def _apply_push_substep(self) -> None:
        if not self.push_scheduled:
            return
        if not self.push_started and self.elapsed_steps >= self.push_start_step:
            self.push_started = True
            self.push_steps_remaining = self.push_duration_steps
        if self.push_steps_remaining <= 0:
            return
        self.data.xfrc_applied[self.pelvis_id, :] = 0.0
        self.data.xfrc_applied[self.pelvis_id, :3] = self.push_force
        self.push_impulse += float(np.linalg.norm(self.push_force) * self.model.opt.timestep)
        self.push_steps_remaining -= 1
    def _reward(self, action, fallen):
        upright_cosine, tilt, pelvis_height = self._upright_metrics()
        angular_velocity = self._sensor("imu_ang_vel")
        raw_contacts, contacts = self._foot_contacts()
        action_delta = action - self.previous_action
        joint_velocity = self._joint_velocities()
        joint_acceleration = abs((joint_velocity - self.previous_joint_velocity) / self.control_dt)
        joint_acceleration_mean = (joint_acceleration + self.joint_acceleration_previous) / 2
        self.joint_acceleration_previous = joint_acceleration
        joint_acceleration_penalty = (self.joint_acceleration_penalty_weight * float(np.sum(joint_acceleration_mean * joint_acceleration_mean)) / 100)
        joint_acceleration_penalty = np.clip(joint_acceleration_penalty, 0, 10)
        # =====================================================
        # CoM und Fußpositionen auslesen (IMMER zuerst!)
        com_pos = self._get_com_position()
        ankle_midpoint = 0.5 * (self._foot_position(0) + self._foot_position(1))
        stance_pos = self._foot_position(self.stance_foot)
        swing_pos = self._foot_position(self.swing_foot)
        com_relative_to_stance = com_pos - stance_pos
        com_relative_to_midpoint = com_pos - ankle_midpoint
        # =====================================================
        # FUSS-ORIENTIERUNG: Standfuß parallel zur Y-Achse
        # Fuß-Orientierung auslesen (Quaternions)
        stance_foot_id = (self.left_ankle_id if self.stance_foot == 0 else self.right_ankle_id)
        stance_foot_quat = self.data.xquat[stance_foot_id].copy()
        # Quaternion in Euler-Winkel umwandeln (roll, pitch, yaw)
        # Wir wollen vor allem den Roll-Winkel (Rotation um X-Achse) kontrollieren
        stance_foot_roll = float(np.arctan2(
            2 * (stance_foot_quat[0] * stance_foot_quat[1] + stance_foot_quat[2] * stance_foot_quat[3]),
            1 - 2 * (stance_foot_quat[1] ** 2 + stance_foot_quat[2] ** 2)))
        # Fuß sollte parallel zur Y-Achse bleiben (Roll ≈ 0)
        stance_foot_parallel_penalty = - 200.0 * stance_foot_roll ** 2

        swing_foot_id = (self.left_ankle_id if self.swing_foot == 0 else self.right_ankle_id)
        swing_foot_quat = self.data.xquat[swing_foot_id].copy()
        swing_foot_roll = float(np.arctan2(
                    2 * (swing_foot_quat[0] * swing_foot_quat[1] + swing_foot_quat[2] * swing_foot_quat[3]),
                    1 - 2 * (swing_foot_quat[1] ** 2 + swing_foot_quat[2] ** 2)))
        swing_foot_parallel_penalty = - 300.0 * swing_foot_roll ** 2
        # =====================================================
        # 4-PHASEN LOGIK: Ein-Bein-Balance
        if self.step_phase == 0:
            com_x_rel = float(com_relative_to_stance[0])
            com_y_rel = float(com_relative_to_stance[1])
            com_x_target = self.com_x_target_phase0
            com_y_target = self.com_y_target_phase0
            x_weight = 0.5
            y_weight = 1.5
        elif self.step_phase == 1:
            com_x_rel = float(com_relative_to_stance[0])
            com_y_rel = float(com_relative_to_stance[1])
            com_x_target = self.com_x_target_phase1
            com_y_target = self.com_y_target_phase1
            x_weight = 0.8
            y_weight = 1.0
        elif self.step_phase == 2:
            com_x_rel = float(com_relative_to_stance[0])
            com_y_rel = float(com_relative_to_stance[1])
            com_x_target = self.com_x_target_phase2
            com_y_target = self.com_y_target_phase2
            x_weight = 1.0
            y_weight = 1.0
        else:  # Phase 3
            com_x_rel = float(com_relative_to_midpoint[0])
            com_y_rel = float(com_relative_to_midpoint[1])
            com_x_target = self.com_x_target_phase3
            com_y_target = self.com_y_target_phase3
            x_weight = 1.5
            y_weight = 1.5
        # CoM-Geschwindigkeit berechnen
        self.com_velocity_x = (com_x_rel - self.previous_com_x_rel) / self.control_dt
        self.com_velocity_y = (com_y_rel - self.previous_com_y_rel) / self.control_dt
        self.previous_com_x_rel = com_x_rel
        self.previous_com_y_rel = com_y_rel
        # Abweichung berechnen
        x_deviation = 10.0 * x_weight * abs(com_x_rel - com_x_target)
        y_deviation = 10.0 * y_weight * abs(com_y_rel - com_y_target)
        # =====================================================
        # CoM RECOVERY BONUS (wieder hinzugefügt!)
        d_x_deviation = self.previous_x_deviation - x_deviation
        self.previous_x_deviation = x_deviation
        d_y_deviation = self.previous_y_deviation - y_deviation
        self.previous_y_deviation = y_deviation
        d_x_deviation_mean = (d_x_deviation + self.previous_dx_deviation) * 0.5
        self.previous_dx_deviation = d_x_deviation
        d_y_deviation_mean = (d_y_deviation + self.previous_dy_deviation) * 0.5
        self.previous_dy_deviation = d_y_deviation
        com_recovery = 0 # np.clip(40 * (d_x_deviation_mean + d_y_deviation_mean),-2,2)
        com_penalty = 0 #-x_deviation - y_deviation
        # =====================================================
        # BALANCE REWARDS
        SURVIVAL_TILT_LIMIT_RAD = np.deg2rad(10.0)
        if tilt >= SURVIVAL_TILT_LIMIT_RAD:
            survival_reward = -tilt
        else:
            survival_reward = (SURVIVAL_TILT_LIMIT_RAD - tilt) * 3.0
        tilt_penalty = 100.0 * tilt ** 2
        height_penalty = 0 # 150.0 * (pelvis_height - self.initial_pelvis_height) ** 2
        angular_penalty = 0 #(0.5 * float(np.sum(angular_velocity * angular_velocity)) if tilt < 0.1 else 0.0)
        #support_reward = 0.05 * float(np.sum(contacts))
        action_rate_penalty = 0 # 0.02 * float(np.sum(action_delta * action_delta))
        recovery_bonus = (40 * (self.previous_tilt - tilt) if tilt < self.previous_tilt else 0.0)
        post_push_bonus = 0 #(0.3 if self.push_started and self.push_steps_remaining <= 0 and tilt < 0.1 and contacts.sum() == 2 else 0.0)
        # =====================================================
        # PHASEN-ÜBERGÄNGE
        swing_dx = float(swing_pos[0] - self.swing_start_pos[0])
        swing_height = float(swing_pos[2] - self.swing_start_pos[2])
        swing_contact = (self._foot_contact_value(self.swing_foot) > self.contact_threshold)        # 1 if swing leg is up
        stance_contact = (self._foot_contact_value(self.stance_foot) > self.contact_threshold)      # 1 if stance foot hasground contact
        # Phase 0
        phase0_ready = (stance_contact and abs(com_y_rel) <= self.com_phase0_tolerance and tilt < 0.40)
        if (abs(com_y_rel) > self.com_phase0_tolerance) or tilt > 0.4:
            self.step_phase = 0
        if self.step_phase == 0:
            if phase0_ready:
                self.phase0_counter += 1
            else:
                self.phase0_counter = 0
            if self.phase0_counter >= self.phase0_hold_steps:
                self.step_phase = 1
                self.phase1_counter = 0
        # Phase 1
        phase1_ready = (swing_height >= self.step_height_target and tilt < 0.4)
        if self.step_phase == 1:
            if phase1_ready:
                self.phase1_counter += 1
            else:
                self.phase1_counter = 0
            if self.phase1_counter >= self.phase1_hold_steps:
                self.step_phase = 2
                self.phase2_counter = 0
        # Phase 2
        phase2_active = (stance_contact and not swing_contact and swing_height >= self.step_height_target and swing_dx >= self.com_x_target_phase2 * self.com_phase2_tolerance and tilt < 0.40)
        if self.step_phase == 2:
            if phase2_active:
                self.phase2_counter += 1
            else:
                self.phase2_counter = 0
            if self.phase2_counter >= self.phase2_min_steps:
                self.step_phase = 3
                self.step_hold_counter = 0
        # Phase 3
        if self.step_phase == 3:
            if stance_contact and swing_contact:
                self.step_hold_counter += 1
            else:
                self.step_hold_counter = 0
        # Episode erfolgreich
        self.step_success = bool(self.step_phase == 3 and self.step_hold_counter >= self.step_hold_steps and tilt < self.max_tilt_rad)
        # =====================================================
        # STEP REWARD
        if self.step_phase == 0:
            COM_P = (0.5 * float(stance_contact) - 20 * abs(com_y_rel))
            Swing_h =  5 * max(swing_height, 0.0)
            self.height_progress_delta = swing_height - self.previous_swing_height
            height_progress_reward = 5 * self.height_progress_delta
            pe = 100 * np.linalg.norm(stance_pos-self.stance_start_pos)
            step_progress_reward = COM_P + Swing_h + height_progress_reward - pe
            self.previous_swing_height = swing_height
        elif self.step_phase == 1:
            COM_P = (0.5 * float(stance_contact) - 20 * abs(com_y_rel))
            Swing_h = 50 * max(swing_height, 0.0)
            self.height_progress_delta = swing_height - self.previous_swing_height
            height_progress_reward = 400 * self.height_progress_delta
            self.previous_swing_height = swing_height
            self.dx_swing_stance = float(swing_pos[0] - stance_pos[0])
            #swing_stance_DX = 10 * self.dx_swing_stance
            pe = 100 * np.linalg.norm(stance_pos-self.stance_start_pos)
            step_progress_reward = COM_P + Swing_h + height_progress_reward - pe# + swing_stance_DX
        elif self.step_phase == 2:
            height_bonus = (3.0 if swing_height >= self.step_height_target else 0.0)
            step_progress_reward = (height_bonus + 0.1 * float(self.phase2_counter))
        else:  # Phase 3
            step_progress_reward = (1.0 * float(stance_contact and swing_contact) - 2.0 * abs(com_x_rel - com_x_target) - 2.0 * abs(com_y_rel - com_y_target))
        # Fußkontakt-Reward
        if self.step_phase in (1, 2):
            foot_contact_reward = 0 #(0.3 * float(stance_contact) - 0.2 * float(swing_contact))
        else:
            foot_contact_reward = 0 #(0.2 * float(stance_contact and swing_contact))
        # Landing Bonus (wieder hinzugefügt!)
        landing_bonus = (2.0 if swing_contact and swing_height < 0.015 and swing_dx >= self.step_target * 0.8 else 0.0)
        step_reward = step_progress_reward + landing_bonus
        step_bonus = 15.0 if self.step_success else 0.0
        # only Sidewalk penalty
        sidewalk_penalty = 2000 *(swing_pos[0] - self.swing_start_pos[0])**2
        # =====================================================
        # GESAMT-REWARD (mit com_recovery!)
        balance_reward = (
                survival_reward
                #+ com_penalty
                - tilt_penalty
                - height_penalty
                - angular_penalty
                #+ support_reward
                #+ foot_contact_reward
                + stance_foot_parallel_penalty
                + swing_foot_parallel_penalty
                - action_rate_penalty
                - joint_acceleration_penalty
                #+ recovery_bonus
                + post_push_bonus
                + 30.0 * self._foot_position(1)[2] - 1
                - sidewalk_penalty
                + com_recovery)
        reward = (balance_reward + step_reward + step_bonus)
        #if fallen:
        #    reward -= 25.0
        self.previous_joint_velocity = joint_velocity.copy()
        self.previous_tilt = tilt
        # =====================================================
        # INFO DICT (vollständig!)
        # =====================================================
        info = {
            #"upright_cosine": upright_cosine,
            #tilt_rad": tilt,
            #"pelvis_height": pelvis_height,
            #"left_contact_force": float(raw_contacts[0]),
            #"right_contact_force": float(raw_contacts[1]),
            #"left_contact": float(contacts[0]),
            #"right_contact": float(contacts[1]),
            #"push_active": float(self.push_steps_remaining > 0),
            #"push_scheduled": float(self.push_scheduled),
            #"push_force_n": float(np.linalg.norm(self.push_force)),
            #"push_impulse_ns": self.push_impulse,
            #"recovery_steps": float(self.recovery_steps),
            #"recovered": float(self.recovered),
            #"reward_survival": survival_reward,
            #"reward_com": com_penalty,
            #"reward_com_x_deviation": -x_deviation,
            #"reward_com_y_deviation": -y_deviation,
            #"reward_tilt_penalty": -tilt_penalty,
            #"reward_support": support_reward,
            #"reward_joint_acceleration_penalty": -joint_acceleration_penalty,
            #"joint_acceleration_rms": float(np.sqrt(np.mean(joint_acceleration * joint_acceleration))),
            #"com_x": float(com_pos[0]),
            #"com_y": float(com_pos[1]),
            #"com_z": float(com_pos[2]),
            #"step_phase": float(self.step_phase),
            #"swing_dx": swing_dx,
            #"swing_height": swing_height,
            #"swing_contact": float(swing_contact),
            #"stance_contact": float(stance_contact),
            #"foot_contact_reward": foot_contact_reward,
            #"step_reward": step_reward,
            #"landing_bonus": landing_bonus,
            #"step_bonus": step_bonus,
            #"step_success": float(self.step_success),
            #"stance_foot_roll": stance_foot_roll,
            #"stance_foot_parallel_penalty": -stance_foot_parallel_penalty,
            #"phase0_counter": float(self.phase0_counter),
            #"phase1_counter": float(self.phase1_counter),
            #"phase2_counter": float(self.phase2_counter),
           # "step_hold_counter": float(self.step_hold_counter),
            "CoM Recovery": com_recovery,}

        plot_values = {
            "reward": reward,
            #"survival": survival_reward,
            #"com_penalty": com_penalty,
            "tilt": -tilt_penalty,
            #"height penalty": -height_penalty,
            #"angular_penalty": -angular_penalty,
            #"support": support_reward,
            #"CoM Recovery": com_recovery,
            #"action_rate": -action_rate_penalty,
            "joint_acc": -joint_acceleration_penalty,
            #"recovery bonus": recovery_bonus,
            #"post_push": post_push_bonus,
            "step_progress_reward": step_progress_reward,
            #"landing_bonus": landing_bonus,
            #"step_bonus": step_bonus,
            "step_phase": self.step_phase,
            "swing_height 100x": 100.0 * swing_height,
            "right_height 100x": 30.0 * self._foot_position(1)[2] - 1,
            #"height_progress_delta 30": 30 * self.height_progress_delta,
            #"dx_swing_stance 1000x": 10 * self.dx_swing_stance,
            "stance_foot_parallel_penalty": stance_foot_parallel_penalty
            }

        if self.reward_writer is None:
            self.reward_writer = csv.DictWriter(
            self.reward_csv,
            fieldnames=["episode","step"] + list(plot_values.keys())
            )
            self.reward_writer.writeheader()

        if self.elapsed_steps % 2 == 0 and self.render_mode == "human":
            row = {
            "episode": self.episode_counter,
            "step": self.elapsed_steps,
            **plot_values
            }
            self.reward_writer.writerow(row)
            self.reward_csv.flush()
            time.sleep(0.05)
            
        return float(reward), info
                        
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Initiale CoM-Position relativ zum Standfuß speichern
        self.previous_swing_height = 0.0
        self.height_progress_delta = 0.0
        com_pos = self._get_com_position()
        stance_pos = self._foot_position(self.stance_foot)
        initial_com_relative = com_pos - stance_pos
        self.dx_swing_stance = 0.0
        self.previous_com_x_rel = float(initial_com_relative[0])
        self.previous_com_y_rel = float(initial_com_relative[1])
        self.com_velocity_x = 0.0
        self.com_velocity_y = 0.0
        self.stance_foot = 0 # int(self.np_random.integers(0, 2))
        self.swing_foot = 1 - self.stance_foot
        self.swing_start_pos = (self._foot_position(self.swing_foot).astype(np.float32))
        self.stance_start_pos = (self._foot_position(self.stance_foot).astype(np.float32))
        self.step_target = float(self.np_random.uniform(0.23, 0.25))
        self.step_height_target = 0.05
        self.step_phase = 0
        self.step_success = False
        self.step_landed = False
        self.step_hold_counter = 0
        self.phase0_counter = 0
        self.phase1_counter = 0
        # CoM-Geschwindigkeit zurücksetzen
        self.previous_com_x_rel = 0.0
        self.previous_com_y_rel = 0.0
        self.com_velocity_x = 0.0
        self.com_velocity_y = 0.0
        self.default_qpos = self._joint_positions()
        self.previous_action[:] = 0.0
        self.previous_joint_velocity = self._joint_velocities()
        self.joint_acceleration_previous = np.zeros(len(ACTUATORS), dtype=np.float32, )
        self.previous_x_deviation = 0.0
        self.previous_y_deviation = 0.0
        self.previous_dx_deviation = 0.0
        self.previous_dy_deviation = 0.0
        self.command[:] = 0.0
        self.initial_pelvis_xy = self.data.xpos[self.pelvis_id, :2].astype(np.float32).copy()
        self.initial_pelvis_height = float(self.data.xpos[self.pelvis_id, 2])
        self.elapsed_steps = 0
        if self.reward_plotter is not None:
            self.reward_plotter.reset()
        self.previous_tilt = 0.0
        self._schedule_push()
        observation = self._observation()
        info = {
            "pelvis_height": self.initial_pelvis_height,
            "control_dt": self.control_dt,
            "push_scheduled": self.push_scheduled,
            "push_start_seconds": self.push_start_step * self.control_dt if self.push_scheduled else None,
            "push_force_n": float(np.linalg.norm(self.push_force)),
        }
        self.episode_counter += 1

        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (len(ACTUATORS),):
            raise ValueError(f"Action must have shape (12,), got {action.shape}")

        applied_action = self._apply_action(action)
        for _ in range(self.frame_skip):
            # In passive mode muss der Python-Code die Viewer-Pause beachten.
            while (
                    self.render_mode == "human"
                    and self.simulation_paused
                    and self.viewer is not None
                    and self.viewer.is_running()
                    #and not getattr(self.viewer, "run", True)
            ):
                self.viewer.sync()
                time.sleep(0.01)
            self.data.qfrc_applied[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            self._apply_push_substep()
            mujoco.mj_step(self.model, self.data)
        self.elapsed_steps += 1
        fallen = self._has_fallen()
        reward, info = self._reward(applied_action, fallen)
        self.previous_action[:] = applied_action
        observation = self._observation()
        terminated = fallen or self.step_success
        truncated = (self.elapsed_steps >= self.max_episode_steps and not terminated)
        if self.render_mode == "human":
            self.render()
        info["elapsed_steps"] = self.elapsed_steps
        return observation, reward, terminated, truncated, info

    def _viewer_key_callback(self, keycode) -> None:
        """Leertaste pausiert bzw. setzt die Simulation fort."""

        if keycode == 32:
            self.simulation_paused = not self.simulation_paused
            state = ("PAUSIERT" if self.simulation_paused else "LÄUFT")
            print(f"Simulation: {state}",flush=True,)
    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._viewer_key_callback,)

        if self.viewer.is_running():
            self.viewer.sync()

    def close(self) -> None:
        if hasattr(self, "reward_csv"):
            self.reward_csv.close()
        # Zusätzliches Matplotlib-Fenster für Reward-Debugging schließen.
        if self.reward_plotter is not None:
            self.reward_plotter.close()
            self.reward_plotter = None
        # Vorhandenes MuJoCo-Viewer-Fenster schließen.
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if self.plot_process is not None:
            self.plot_process.terminate()
            try:
                self.plot_process.wait(timeout=1)
            except Exception:
                pass
            self.plot_process = None

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = AsimovWalkEnv(
        DEFAULT_MODEL,
        render_mode=None,
        push_enabled=True,
        push_probability=1.0,
        min_push_force=50.0,
        max_push_force=50.0,
        push_start_min=0.2,
        push_start_max=0.2,
        push_duration_min=0.08,
        push_duration_max=0.08,
    )
    check_env(env, warn=True)
    observation, info = env.reset(seed=42)
    print("Environment check: OK")
    print("Observation shape:", observation.shape)
    print("Action shape:", env.action_space.shape)
    print("Push scheduled:", info["push_scheduled"])
    env.close()
