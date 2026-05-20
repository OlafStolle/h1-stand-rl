"""Watch a trained H1 standing policy live in the MuJoCo viewer."""

import glob
import os
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from h1_stand_env import H1StandEnv, FRAME_SKIP

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")


def _find_checkpoint():
    final = os.path.join(CHECKPOINT_DIR, "h1_stand_final.zip")
    if os.path.exists(final):
        return final
    candidates = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    return max(candidates, key=os.path.getmtime)


def main():
    ckpt = _find_checkpoint()
    print(f"Loading policy: {ckpt}")
    model = PPO.load(ckpt)

    env = H1StandEnv()
    obs, _ = env.reset()

    dt = env.model.opt.timestep * FRAME_SKIP

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            if terminated or truncated:
                obs, _ = env.reset()
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
