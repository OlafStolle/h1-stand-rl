"""Headless evaluation: measure objectively whether the H1 learned to stand."""

import argparse
import glob
import os

import numpy as np
from stable_baselines3 import PPO

from h1_stand_env import H1StandEnv, MAX_EPISODE_STEPS

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")

# --- Verdict thresholds ---
STAND_SUCCESS_RATE = 0.80   # >= 80% of episodes must reach full horizon
STAND_HEIGHT = 0.85         # mean torso height (last 200 steps) must exceed this
TAIL_WINDOW = 200           # steps averaged for the "settled" height


def _find_checkpoint():
    """Prefer h1_stand_final.zip, else newest .zip in checkpoints/."""
    final = os.path.join(CHECKPOINT_DIR, "h1_stand_final.zip")
    if os.path.exists(final):
        return final
    candidates = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    return max(candidates, key=os.path.getmtime)


def _run_episode(model, env):
    """Run one deterministic episode; return per-episode metrics."""
    obs, _ = env.reset()
    qpos_start = env.data.qpos.copy()
    heights = []
    qvels_lower = []
    qvels_upper = []
    truncated = False
    terminated = False
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        heights.append(info["height"])
        # Collect joint velocities (indices 6:25 = 19 joints; 0-9 = lower, 10-18 = upper)
        qvel = env.data.qvel[6:25]
        qvels_lower.append(np.abs(qvel[:10]).mean())
        qvels_upper.append(np.abs(qvel[10:19]).mean())
        steps += 1
        if terminated or truncated:
            break
    tail = heights[-TAIL_WINDOW:]

    # Stillness metrics
    qv_lower = float(np.mean(qvels_lower))
    qv_upper = float(np.mean(qvels_upper))
    qv_all = (qv_lower * 10 + qv_upper * 9) / 19

    # XY drift (horizontal displacement from start to end)
    qpos_end = env.data.qpos.copy()
    xy_drift = float(np.hypot(qpos_end[0] - qpos_start[0], qpos_end[1] - qpos_start[1]))

    # Height std
    h_std = float(np.std(heights))

    return {
        "steps": steps,
        "truncated": truncated,
        "terminated": terminated,
        "final_height": heights[-1],
        "tail_mean_height": float(np.mean(tail)),
        "qv_lower": qv_lower,
        "qv_upper": qv_upper,
        "qv_all": qv_all,
        "xy_drift": xy_drift,
        "h_std": h_std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained H1 standing policy (headless)."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    ckpt = args.checkpoint or _find_checkpoint()
    print(f"Checkpoint : {ckpt}")
    print(f"Episodes   : {args.episodes}\n")

    model = PPO.load(ckpt)
    env = H1StandEnv()

    results = [_run_episode(model, env) for _ in range(args.episodes)]
    env.close()

    lengths = np.array([r["steps"] for r in results])
    full_runs = np.array([r["truncated"] for r in results])
    final_heights = np.array([r["final_height"] for r in results])
    tail_heights = np.array([r["tail_mean_height"] for r in results])
    qv_lowers = np.array([r["qv_lower"] for r in results])
    qv_uppers = np.array([r["qv_upper"] for r in results])
    qv_alls = np.array([r["qv_all"] for r in results])
    xy_drifts = np.array([r["xy_drift"] for r in results])
    h_stds = np.array([r["h_std"] for r in results])

    success_rate = float(np.mean(full_runs))
    mean_tail_height = float(np.mean(tail_heights))

    # --- Per-episode table ---
    print(f"{'Ep':>3} | {'Steps':>5} | {'Outcome':>10} | "
          f"{'FinalH':>7} | {'TailH':>7} | {'qv_l':>7} | {'qv_u':>7} | {'xy_dr':>7}")
    print("-" * 80)
    for i, r in enumerate(results):
        outcome = "FULL 1000" if r["truncated"] else "fell"
        print(f"{i + 1:>3} | {r['steps']:>5} | {outcome:>10} | "
              f"{r['final_height']:>7.3f} | {r['tail_mean_height']:>7.3f} | "
              f"{r['qv_lower']:>7.4f} | {r['qv_upper']:>7.4f} | {r['xy_drift']:>7.3f}")

    # --- Aggregate block ---
    print("\n=== Aggregate ===")
    print(f"Episode length   mean/min/max : "
          f"{lengths.mean():.1f} / {lengths.min()} / {lengths.max()}")
    print(f"Full-horizon ({MAX_EPISODE_STEPS} steps) reached : "
          f"{success_rate * 100:.1f} %")
    print(f"Final torso height       mean : {final_heights.mean():.3f} m")
    print(f"Tail torso height ({TAIL_WINDOW}st) mean : {mean_tail_height:.3f} m")
    print(f"Lower-body qvel (legs)   mean : {qv_lowers.mean():.4f} rad/s")
    print(f"Upper-body qvel (arms)   mean : {qv_uppers.mean():.4f} rad/s")
    print(f"Weighted qvel (all)      mean : {qv_alls.mean():.4f} rad/s")
    print(f"XY drift (horiz disp)    mean : {xy_drifts.mean():.3f} m")
    print(f"Height std               mean : {h_stds.mean():.3f} m")

    # --- Verdict ---
    pass_rate = success_rate >= STAND_SUCCESS_RATE
    pass_height = mean_tail_height > STAND_HEIGHT
    print("\n=== Verdict ===")
    if pass_rate and pass_height:
        print("STANDS")
    else:
        reasons = []
        if not pass_rate:
            reasons.append(
                f"only {success_rate * 100:.1f}% full episodes "
                f"(>= {STAND_SUCCESS_RATE * 100:.0f}% required)"
            )
        if not pass_height:
            reasons.append(
                f"tail height {mean_tail_height:.3f} m "
                f"(> {STAND_HEIGHT} m required)"
            )
        print("NOT STANDING YET -- " + "; ".join(reasons))


if __name__ == "__main__":
    main()
