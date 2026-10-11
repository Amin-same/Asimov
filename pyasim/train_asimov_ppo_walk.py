#!/usr/bin/env python3
"""Asimov PPO training backend for the selectable-joints training GUI.

The base environment always receives a complete 12-element action vector.
PPO only receives actions for the selected actuator indices. All unselected
actuators are fixed to the corresponding value in --fixed-action (default 0).
A metadata JSON file is saved next to every final model for reliable evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from asimov_walk_env import ACTUATORS, AsimovWalkEnv


class SelectedActuatorWrapper(gym.Wrapper):
    """Expose PPO actions only for selected actuators, expand to 12 actions."""

    def __init__(
        self,
        env: gym.Env,
        active_indices: np.ndarray,
        fixed_action: np.ndarray,
    ) -> None:
        super().__init__(env)
        self.active_indices = np.asarray(active_indices, dtype=np.int64)
        self.fixed_action = np.asarray(fixed_action, dtype=np.float32).copy()

        if self.active_indices.ndim != 1 or self.active_indices.size == 0:
            raise ValueError("At least one active actuator must be selected.")
        if np.any(self.active_indices < 0) or np.any(
            self.active_indices >= len(ACTUATORS)
        ):
            raise ValueError(
                f"Active actuator indices must be between 0 and {len(ACTUATORS) - 1}."
            )
        if np.unique(self.active_indices).size != self.active_indices.size:
            raise ValueError("Active actuator indices must be unique.")
        if self.fixed_action.shape != (len(ACTUATORS),):
            raise ValueError(
                f"fixed_action must contain {len(ACTUATORS)} values, "
                f"got {self.fixed_action.shape}."
            )
        if not np.all(np.isfinite(self.fixed_action)):
            raise ValueError("fixed_action contains a non-finite value.")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.active_indices.size,),
            dtype=np.float32,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected PPO action shape {self.action_space.shape}, got {action.shape}."
            )

        full_action = self.fixed_action.copy()
        full_action[self.active_indices] = np.clip(action, -1.0, 1.0)
        return self.env.step(full_action)


def parse_indices(text: str) -> np.ndarray:
    try:
        values = np.asarray(
            [int(item.strip()) for item in text.split(",") if item.strip()],
            dtype=np.int64,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "active-indices must be comma-separated integer indices."
        ) from exc

    if values.size == 0:
        raise argparse.ArgumentTypeError("Select at least one active actuator.")
    if np.any(values < 0) or np.any(values >= len(ACTUATORS)):
        raise argparse.ArgumentTypeError(
            f"Active indices must be between 0 and {len(ACTUATORS) - 1}."
        )
    if np.unique(values).size != values.size:
        raise argparse.ArgumentTypeError("Active indices must not contain duplicates.")
    return values


def parse_12_values(text: str, argument_name: str) -> np.ndarray:
    try:
        values = np.asarray(
            [float(item.strip()) for item in text.split(",")],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must be a comma-separated list of numbers."
        ) from exc

    if values.shape != (len(ACTUATORS),):
        raise argparse.ArgumentTypeError(
            f"{argument_name} must contain exactly {len(ACTUATORS)} values, "
            f"got {values.size}."
        )
    if not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            f"{argument_name} must contain only finite values."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO for Asimov walk."
    )

    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--run-name", type=str, default="asimov_walk_selectable")
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path(__file__).parent / "asimov0.xml",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--visualize", action="store_true")

    parser.add_argument("--max-push-force", type=float, default=4.0)
    parser.add_argument("--min-push-force", type=float, default=2.0)
    parser.add_argument("--push-prob", type=float, default=0.8)
    parser.add_argument("--push-start-min", type=float, default=2.0)
    parser.add_argument("--push-start-max", type=float, default=5.0)
    parser.add_argument("--push-duration-min", type=float, default=0.2)
    parser.add_argument("--push-duration-max", type=float, default=0.5)

    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--max-episode-seconds", type=float, default=6.0)
    parser.add_argument("--contact-threshold", type=float, default=5.0)
    parser.add_argument("--pelvis-min-height", type=float, default=0.35)
    parser.add_argument("--max-tilt-rad", type=float, default=0.70)

    parser.add_argument(
        "--active-indices",
        type=parse_indices,
        default=np.arange(len(ACTUATORS), dtype=np.int64),
        help="Comma-separated selected actuator indices, e.g. 3,4,5,9,10,11",
    )
    parser.add_argument(
        "--kp-values",
        type=lambda text: parse_12_values(text, "kp-values"),
        default=np.full(len(ACTUATORS), 60.0, dtype=np.float64),
        help="Exactly 12 comma-separated Kp values.",
    )
    parser.add_argument(
        "--kv-values",
        type=lambda text: parse_12_values(text, "kv-values"),
        default=np.full(len(ACTUATORS), 6.0, dtype=np.float64),
        help="Exactly 12 comma-separated Kv values.",
    )
    parser.add_argument(
        "--fixed-action",
        type=lambda text: parse_12_values(text, "fixed-action"),
        default=np.zeros(len(ACTUATORS), dtype=np.float64),
        help="Exactly 12 values for inactive actuators; defaults to all zero.",
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)

    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive.")
    if args.n_envs <= 0:
        parser.error("--n-envs must be positive.")
    if args.n_steps <= 0 or args.batch_size <= 0 or args.n_epochs <= 0:
        parser.error("--n-steps, --batch-size and --n-epochs must be positive.")
    if not 0.0 <= args.push_prob <= 1.0:
        parser.error("--push-prob must be between 0 and 1.")
    if args.min_push_force < 0.0 or args.max_push_force < args.min_push_force:
        parser.error("Push force limits are invalid.")
    if np.any(args.kp_values < 0.0) or np.any(args.kv_values < 0.0):
        parser.error("All Kp and Kv values must be non-negative.")

    return args


def make_env(
    model_xml: Path,
    max_push_force: float,
    min_push_force: float,
    push_prob: float,
    push_start_min: float,
    push_start_max: float,
    push_duration_min: float,
    push_duration_max: float,
    seed: int,
    render_mode: Optional[str],
    active_indices: np.ndarray,
    kp_values: np.ndarray,
    kv_values: np.ndarray,
    fixed_action: np.ndarray,
    action_scale: float,
    max_episode_seconds: float,
    contact_threshold: float,
    pelvis_min_height: float,
    max_tilt_rad: float,
):
    def _init():
        env = AsimovWalkEnv(
            model_path=model_xml,
            render_mode=render_mode,
            control_dt=0.02,
            action_scale=action_scale,
            max_episode_seconds=max_episode_seconds,
            contact_threshold=contact_threshold,
            pelvis_min_height=pelvis_min_height,
            max_tilt_rad=max_tilt_rad,
            kp=kp_values,
            kv=kv_values,

            # Für Phase 1 unbedingt zuerst ausschalten
            push_enabled=False,
            push_probability=0.0,

            min_push_force=min_push_force,
            max_push_force=max_push_force,
            push_start_min=push_start_min,
            push_start_max=push_start_max,
            push_duration_min=push_duration_min,
            push_duration_max=push_duration_max,
        )

        env = SelectedActuatorWrapper(
            env,
            active_indices=active_indices,
            fixed_action=fixed_action,
        )
        env.reset(seed=seed)
        return env

    return _init


def save_metadata(model_dir: Path, args: argparse.Namespace) -> Path:
    metadata = {
        "environment": "AsimovWalkEnv",
        "task": "single_step",
        "step_target": 0.20,
        "actuators_full_order": list(ACTUATORS),
        "active_indices": args.active_indices.tolist(),
        "active_actuators": [ACTUATORS[index] for index in args.active_indices],
        "fixed_action": np.asarray(args.fixed_action, dtype=float).tolist(),
        "kp_values": np.asarray(args.kp_values, dtype=float).tolist(),
        "kv_values": np.asarray(args.kv_values, dtype=float).tolist(),
        "action_scale": float(args.action_scale),
        "control_dt": 0.02,
        "observation_size": 58,
    }
    metadata_path = model_dir / "ppo_asimov_walk_final_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    args.model_xml = args.model_xml.resolve()
    if not args.model_xml.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {args.model_xml}")

    run_dir = Path(__file__).parent / "runs" / args.run_name
    log_dir = run_dir / "logs"
    model_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    render_mode = "human" if args.visualize else None
    n_envs = 1 if args.visualize else args.n_envs

    print("=" * 72)
    print("Asimov PPO training: selectable actuators + walk")
    print("=" * 72)
    print(f"Model XML       : {args.model_xml}")
    print(f"Run directory   : {run_dir}")
    print(f"Training steps  : {args.timesteps:,}")
    print("Observation     : 58 values")
    print(f"PPO actions     : {len(args.active_indices)} selected values")
    print(f"Active indices  : {args.active_indices.tolist()}")
    print("Active joints   : " + ", ".join(ACTUATORS[index] for index in args.active_indices))
    print(f"Fixed action    : {np.asarray(args.fixed_action, dtype=float).tolist()}")
    print(f"Number envs     : {n_envs}")
    print(f"Visualization   : {render_mode}")
    print("=" * 72)

    env_fn = make_env(
        model_xml=args.model_xml,
        max_push_force=args.max_push_force,
        min_push_force=args.min_push_force,
        push_prob=args.push_prob,
        push_start_min=args.push_start_min,
        push_start_max=args.push_start_max,
        push_duration_min=args.push_duration_min,
        push_duration_max=args.push_duration_max,
        seed=args.seed,
        render_mode=render_mode,
        active_indices=args.active_indices,
        kp_values=args.kp_values,
        kv_values=args.kv_values,
        fixed_action=args.fixed_action,
        action_scale=args.action_scale,
        max_episode_seconds=args.max_episode_seconds,
        contact_threshold=args.contact_threshold,
        pelvis_min_height=args.pelvis_min_height,
        max_tilt_rad=args.max_tilt_rad,
    )

    env = make_vec_env(
        env_fn,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=None,
    )

    try:
        print("Gymnasium/SB3 environment check: OK")
        print("PPO action space:", env.action_space)

        if args.resume is not None:
            resume_path = args.resume.resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume model not found: {resume_path}")
            print(f"Resuming PPO model: {resume_path}")
            model = PPO.load(
                str(resume_path),
                env=env,
                device="auto",
                tensorboard_log=str(log_dir),
            )
        else:
            print("Training new PPO model from scratch")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                verbose=1,
                tensorboard_log=str(log_dir),
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, 25_000 // n_envs),
            save_path=str(model_dir),
            name_prefix="ppo_asimov_walk",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
        )

        final_model_path = model_dir / "ppo_asimov_walk_final.zip"
        model.save(str(final_model_path))
        metadata_path = save_metadata(model_dir, args)
        print(f"Saved final PPO model: {final_model_path}")
        print(f"Saved model metadata : {metadata_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
