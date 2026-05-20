"""PPO training for the H1 standing-balance task."""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from h1_stand_env import H1StandEnv
from h1_standup_env import H1StandupEnv
from record_callback import RecordCallback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
TB_DIR = os.path.join(_THIS_DIR, "tb")

ENV_CLASSES = {"stand": H1StandEnv, "standup": H1StandupEnv}


def _make_stand_env():
    return H1StandEnv()


def _make_standup_env():
    return H1StandupEnv()


_ENV_FACTORIES = {"stand": _make_stand_env, "standup": _make_standup_env}


def main():
    parser = argparse.ArgumentParser(description="Train H1 to stand with PPO.")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--record-freq", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--env",
        type=str,
        default="stand",
        choices=["stand", "standup"],
        help="Which environment: 'stand' (Phase 1) or 'standup' (Phase 2).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .zip to resume from. With --resume, "
        "--timesteps means the ADDITIONAL steps to train.",
    )
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    env_factory = _ENV_FACTORIES[args.env]
    vec_env = SubprocVecEnv([env_factory for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    if args.resume:
        model = PPO.load(
            args.resume,
            env=vec_env,
            device="cpu",
            custom_objects={"learning_rate": args.learning_rate},
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=2048,
            batch_size=512,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            learning_rate=args.learning_rate,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=TB_DIR,
            verbose=1,
        )

    callbacks = [
        RecordCallback(record_freq=args.record_freq, verbose=1),
        CheckpointCallback(
            save_freq=max(500_000 // args.n_envs, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="h1_stand",
        ),
    ]

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
    )
    model.save(os.path.join(CHECKPOINT_DIR, "h1_stand_final"))
    vec_env.close()


if __name__ == "__main__":
    main()
