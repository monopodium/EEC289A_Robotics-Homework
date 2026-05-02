#!/usr/bin/env python3
"""Generate figures used in the homework report.

Reads:
- artifacts/per_direction_eval/per_direction.json
- artifacts/per_direction_eval/timeseries.npz
- artifacts/public_eval_bundle/public_eval.json
- artifacts/run_baseline/stage_1/progress.json
- artifacts/run_baseline/stage_2/progress.json

Writes PNGs to <output-dir> (default: artifacts/figures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


DIR_LABEL = {
    "plus_vx":   "+vx",
    "minus_vx":  "-vx",
    "plus_vy":   "+vy",
    "minus_vy":  "-vy",
    "plus_yaw":  "+yaw",
    "minus_yaw": "-yaw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "figures")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r") as handle:
        return json.load(handle)


def plot_training_curves(artifacts_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for stage_name, color in (("stage_1", "tab:blue"), ("stage_2", "tab:orange")):
        progress_path = artifacts_dir / "run_baseline" / stage_name / "progress.json"
        records = load_json(progress_path)
        if not records:
            continue
        steps = []
        rewards = []
        rew_std = []
        for record in records:
            metrics = record.get("metrics", {})
            reward = metrics.get("eval/episode_reward")
            if reward is None:
                continue
            steps.append(record["num_steps"])
            rewards.append(float(reward))
            rew_std.append(float(metrics.get("eval/episode_reward_std", 0.0)))
        if not steps:
            continue
        steps = np.asarray(steps)
        rewards = np.asarray(rewards)
        rew_std = np.asarray(rew_std)
        ax.errorbar(steps, rewards, yerr=rew_std, label=stage_name, color=color, linewidth=2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval episode reward")
    ax.set_title("PPO training curves (Go2 joystick)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)


def plot_per_direction_bar(artifacts_dir: Path, out_dir: Path) -> None:
    data = load_json(artifacts_dir / "per_direction_eval" / "per_direction.json")
    if data is None:
        return
    results = data["results"]

    by_axis: dict[str, list[tuple[float, float, float]]] = {}
    for record in results:
        axis = record["axis"]
        if axis == "combined":
            continue
        by_axis.setdefault(axis, []).append(
            (record["magnitude"], record["mean_lin_tracking_err"], record["mean_yaw_tracking_err"])
        )

    fig, (ax_lin, ax_yaw) = plt.subplots(1, 2, figsize=(12, 4))
    axes_in_order = ["plus_vx", "minus_vx", "plus_vy", "minus_vy", "plus_yaw", "minus_yaw"]
    bar_width = 0.27

    for plot_axis, idx in (("lin", 1), ("yaw", 2)):
        ax = ax_lin if plot_axis == "lin" else ax_yaw
        for k, magnitude in enumerate([0.3, 0.6, 0.9]):
            heights = []
            xs = []
            for i, axis in enumerate(axes_in_order):
                rows = by_axis.get(axis, [])
                row = next((r for r in rows if abs(r[0] - magnitude) < 1e-6), None)
                if row is None:
                    continue
                heights.append(row[idx])
                xs.append(i + (k - 1) * bar_width)
            ax.bar(xs, heights, width=bar_width, label=f"|cmd|={magnitude:.1f}")
        ax.set_xticks(range(len(axes_in_order)))
        ax.set_xticklabels([DIR_LABEL[a] for a in axes_in_order])
        ax.set_ylabel("mean tracking error")
        ax.set_title(f"{'linear' if plot_axis == 'lin' else 'yaw-rate'} tracking error per direction")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "per_direction_bars.png", dpi=130)
    plt.close(fig)


def plot_timeseries(artifacts_dir: Path, out_dir: Path) -> None:
    npz_path = artifacts_dir / "per_direction_eval" / "timeseries.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path)
    interesting_tests = [
        ("plus_vx_mag_0.60", 0, "vx (m/s)"),
        ("minus_vx_mag_0.60", 0, "vx (m/s)"),
        ("plus_vy_mag_0.30", 1, "vy (m/s)"),
        ("plus_yaw_mag_0.60", 2, "yaw_rate (rad/s)"),
        ("minus_yaw_mag_0.60", 2, "yaw_rate (rad/s)"),
        ("vx_plus_yaw", 0, "vx (m/s)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    axes_flat = list(axes.flat)

    for ax, (test, axis_idx, label) in zip(axes_flat, interesting_tests):
        lin_key = f"{test}__lin"
        yaw_key = f"{test}__yaw"
        if lin_key not in data:
            ax.set_title(f"missing: {test}")
            continue
        lin = data[lin_key]
        yaw = data[yaw_key]
        t = np.arange(lin.shape[0]) * 0.02
        if axis_idx in (0, 1):
            measured = lin[:, axis_idx]
            cmd_value = float(lin[0, axis_idx])  # Approximation: not used directly.
        else:
            measured = yaw
        ax.plot(t, measured, color="tab:blue", label="measured")
        ax.set_title(test)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    if axes_flat:
        axes_flat[-1].set_xlabel("time (s)")
    if len(axes_flat) >= 2:
        axes_flat[-2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "timeseries.png", dpi=130)
    plt.close(fig)


def plot_public_eval_per_episode(artifacts_dir: Path, out_dir: Path) -> None:
    eval_json = load_json(artifacts_dir / "public_eval_bundle" / "public_eval.json")
    if eval_json is None:
        return
    summaries = eval_json.get("per_episode_summary", [])
    if not summaries:
        return
    labels = [s.get("episode_label", str(s["episode_id"])) for s in summaries]
    lin_err = [s["velocity_tracking_error"] for s in summaries]
    yaw_err = [s["yaw_tracking_error"] for s in summaries]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, lin_err, width=0.35, label="lin tracking err")
    ax.bar(x + 0.18, yaw_err, width=0.35, label="yaw tracking err")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("mean tracking error")
    ax.set_title("public benchmark per-episode error")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "public_eval_per_episode.png", dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(args.artifacts_dir, out_dir)
    plot_per_direction_bar(args.artifacts_dir, out_dir)
    plot_timeseries(args.artifacts_dir, out_dir)
    plot_public_eval_per_episode(args.artifacts_dir, out_dir)
    print("[make_report_figures] wrote figures to", out_dir)


if __name__ == "__main__":
    main()
