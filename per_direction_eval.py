#!/usr/bin/env python3
"""Per-direction evaluation that complements public_eval.py.

For each canonical direction (+vx, -vx, +vy, -vy, +yaw, -yaw) the script
holds a constant command for `--seconds-per-test` seconds and records:
- mean command vs. measured velocity / yaw
- absolute tracking error along the active axis
- whether the robot fell during the test
- per-step time series of cmd vs measured (for plotting)

A sweep of magnitudes is run for each direction so we can plot a
tracking-error vs. command-magnitude curve. Combined commands probe
simultaneous-axis tracking. Results land in
`<output-dir>/per_direction.{json,csv}` and `<output-dir>/timeseries.npz`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent


# (label, axis_index, sign) where axis_index is 0=vx, 1=vy, 2=yaw_rate.
DIRECTIONS = (
    ("plus_vx",   0,  1.0),
    ("minus_vx",  0, -1.0),
    ("plus_vy",   1,  1.0),
    ("minus_vy",  1, -1.0),
    ("plus_yaw",  2,  1.0),
    ("minus_yaw", 2, -1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--seconds-per-test", type=float, default=8.0)
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[0.3, 0.6, 0.9],
        help="Per-axis magnitudes to sweep (interpreted as absolute values).",
    )
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def _force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def run_one(env, policy, jax, reset_fn, step_fn, command: np.ndarray, num_steps: int, seed: int) -> dict[str, Any]:
    rng = jax.random.PRNGKey(seed)
    state = reset_fn(rng)
    state = _force_command(state, command, jax)

    measured_lin = []
    measured_yaw = []
    fell = False
    done_step = None

    for step_idx in range(num_steps):
        rng, act_key = jax.random.split(rng)
        action, _ = policy(state.obs, act_key)
        state = step_fn(state, action)
        state = _force_command(state, command, jax)

        measured_lin.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
        measured_yaw.append(float(np.asarray(env.get_gyro(state.data)[2])))

        if bool(np.asarray(state.done)) and not fell:
            fell = True
            done_step = step_idx + 1
            break

    measured_lin = np.asarray(measured_lin, dtype=np.float32)
    measured_yaw = np.asarray(measured_yaw, dtype=np.float32)
    cmd_lin = np.asarray(command[:2], dtype=np.float32)
    cmd_yaw = float(command[2])

    return {
        "command": [float(x) for x in command],
        "num_steps": int(measured_lin.shape[0]),
        "fell": bool(fell),
        "done_step": done_step,
        "mean_measured_vx": float(np.mean(measured_lin[:, 0])) if measured_lin.size else 0.0,
        "mean_measured_vy": float(np.mean(measured_lin[:, 1])) if measured_lin.size else 0.0,
        "mean_measured_yaw_rate": float(np.mean(measured_yaw)) if measured_yaw.size else 0.0,
        "mean_lin_tracking_err": float(np.mean(np.linalg.norm(cmd_lin - measured_lin, axis=-1)))
        if measured_lin.size
        else float("nan"),
        "mean_yaw_tracking_err": float(np.mean(np.abs(cmd_yaw - measured_yaw)))
        if measured_yaw.size
        else float("nan"),
        "_measured_lin_series": measured_lin,
        "_measured_yaw_series": measured_yaw,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True

    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)

    num_steps = int(round(args.seconds_per_test / env_cfg.ctrl_dt))
    env_cfg.episode_length = max(num_steps + 50, env_cfg.episode_length)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    results: list[dict[str, Any]] = []
    timeseries: dict[str, np.ndarray] = {}
    seed = int(config["seed"]) + 17

    for label, axis, sign in DIRECTIONS:
        for magnitude in args.magnitudes:
            command = np.zeros(3, dtype=np.float32)
            command[axis] = sign * float(magnitude)
            record = run_one(env, policy, jax, reset_fn, step_fn, command, num_steps, seed)
            test_label = f"{label}_mag_{magnitude:.2f}"
            timeseries[f"{test_label}__lin"] = record.pop("_measured_lin_series")
            timeseries[f"{test_label}__yaw"] = record.pop("_measured_yaw_series")
            record["test_label"] = test_label
            record["axis"] = label
            record["magnitude"] = float(magnitude)
            print(json.dumps(record, indent=2))
            results.append(record)
            seed += 1

    combined_tests = [
        ("vx_plus_yaw", np.array([0.6, 0.0, 0.6], dtype=np.float32)),
        ("vx_plus_vy_plus_yaw", np.array([0.6, 0.3, 0.6], dtype=np.float32)),
        ("minus_vx_minus_yaw", np.array([-0.6, 0.0, -0.6], dtype=np.float32)),
    ]
    for label, command in combined_tests:
        record = run_one(env, policy, jax, reset_fn, step_fn, command, num_steps, seed)
        timeseries[f"{label}__lin"] = record.pop("_measured_lin_series")
        timeseries[f"{label}__yaw"] = record.pop("_measured_yaw_series")
        record["test_label"] = label
        record["axis"] = "combined"
        record["magnitude"] = float(np.linalg.norm(command))
        print(json.dumps(record, indent=2))
        results.append(record)
        seed += 1

    save_json(output_dir / "per_direction.json", {"results": results})
    np.savez(output_dir / "timeseries.npz", **timeseries)

    csv_path = output_dir / "per_direction.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_label",
                "axis",
                "magnitude",
                "cmd_vx",
                "cmd_vy",
                "cmd_yaw",
                "mean_measured_vx",
                "mean_measured_vy",
                "mean_measured_yaw_rate",
                "mean_lin_tracking_err",
                "mean_yaw_tracking_err",
                "fell",
            ]
        )
        for record in results:
            writer.writerow(
                [
                    record["test_label"],
                    record["axis"],
                    f"{record['magnitude']:.3f}",
                    f"{record['command'][0]:.3f}",
                    f"{record['command'][1]:.3f}",
                    f"{record['command'][2]:.3f}",
                    f"{record['mean_measured_vx']:.4f}",
                    f"{record['mean_measured_vy']:.4f}",
                    f"{record['mean_measured_yaw_rate']:.4f}",
                    f"{record['mean_lin_tracking_err']:.4f}",
                    f"{record['mean_yaw_tracking_err']:.4f}",
                    "1" if record["fell"] else "0",
                ]
            )
    print(f"\n[per_direction_eval] wrote {csv_path}")


if __name__ == "__main__":
    main()
