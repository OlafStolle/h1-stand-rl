"""SB3 callback: render a greedy policy rollout to MP4 at fixed intervals."""

import os

# Configure offscreen GL before any MuJoCo rendering import path is hit.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import imageio
from stable_baselines3.common.callbacks import BaseCallback

from h1_stand_env import H1StandEnv, MAX_EPISODE_STEPS

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
VIDEO_FPS = 30


def _make_render_env():
    """Create a render-capable env, falling back to osmesa if egl fails."""
    try:
        return H1StandEnv(render_mode="rgb_array")
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        return H1StandEnv(render_mode="rgb_array")


class RecordCallback(BaseCallback):
    """Records a deterministic rollout every `record_freq` training steps."""

    def __init__(self, record_freq=200_000, verbose=0):
        super().__init__(verbose)
        self.record_freq = record_freq
        self._last_record = 0

    def _on_training_start(self):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self._record(0)  # untrained baseline

    def _on_step(self):
        if self.num_timesteps - self._last_record >= self.record_freq:
            self._last_record = self.num_timesteps
            self._record(self.num_timesteps)
        return True

    def _record(self, steps):
        env = _make_render_env()
        try:
            obs, _ = env.reset()
            frames = []
            for _ in range(MAX_EPISODE_STEPS):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                frames.append(env.render())
                if terminated or truncated:
                    break
            path = os.path.join(VIDEO_DIR, f"versuch_{steps:07d}.mp4")
            imageio.mimsave(path, frames, fps=VIDEO_FPS)
            if self.verbose:
                print(f"[RecordCallback] saved {path} ({len(frames)} frames)")
        finally:
            env.close()
