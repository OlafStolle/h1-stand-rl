# h1-stand-rl — Unitree H1 humanoid learns to stand via PPO + MuJoCo

![Algorithm: PPO](https://img.shields.io/badge/algorithm-PPO-blue)
![Simulator: MuJoCo](https://img.shields.io/badge/simulator-MuJoCo%203.8-green)
![Library: Stable-Baselines3](https://img.shields.io/badge/library-Stable--Baselines3%202.8-orange)

A Unitree H1 humanoid robot learns to maintain a standing balance entirely in simulation.
Starting from 0% success, the agent reaches **97% success on 10-second standing episodes**
through 6 reward-engineering iterations driven by human domain knowledge.

---

## What this is

Phase 1 of a humanoid locomotion project: teach a 19-DOF bipedal robot to stand still.

The robot starts in the `home` keyframe (upright pose) with small random noise.
It must stay standing for 1000 control steps (~10 seconds) without falling.
A fall is defined as torso height dropping below 0.5 m.

What makes this non-trivial: naive reward functions (height + uprightness only) produce
a wildly oscillating robot that never converges. Six iterations of human-guided reward
engineering — drawing on biomechanics, motor-control heuristics, and clip_fraction
diagnostics — were needed to reach a stable, statue-like stand.

---

## Quick start

### Prerequisites

- Python 3.11
- MuJoCo system libraries (see [MuJoCo installation](https://mujoco.readthedocs.io/en/stable/programming/index.html))
- ffmpeg (for video recording)

### Install

```bash
git clone https://github.com/OlafStolle/h1-stand-rl.git
cd h1-stand-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU users:** replace the torch line in `requirements.txt` with a CUDA wheel before installing.
> See comment at the top of `requirements.txt`.

### MuJoCo Menagerie (required)

The Unitree H1 model is not included. Clone it one level up:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie ../mujoco_menagerie
```

The scene path is resolved relative to this repo: `../mujoco_menagerie/unitree_h1/scene.xml`.

### Train

```bash
python train.py --timesteps 25000000 --n-envs 14
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | 3 000 000 | Total training steps |
| `--n-envs` | 12 | Parallel environments (SubprocVecEnv) |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--record-freq` | 200 000 | Steps between video snapshots |
| `--resume PATH` | — | Resume from checkpoint; `--timesteps` = additional steps |
| `--env` | `stand` | `stand` (Phase 1) or `standup` (Phase 2, experimental) |

Checkpoints are saved to `checkpoints/` every 500k steps and at the end as `h1_stand_final.zip`.

### Watch (live viewer)

```bash
python watch.py
```

Opens the MuJoCo interactive viewer. Loads the latest checkpoint automatically.

### Evaluate (headless)

```bash
python evaluate.py --episodes 30
```

Prints a per-episode table and aggregate metrics (success rate, tail height, joint velocities, XY drift).

---

## Project layout

| File / Folder | Description |
|---------------|-------------|
| `h1_stand_env.py` | Phase 1 environment — standing balance, 19 torque actuators |
| `h1_standup_env.py` | Phase 2 environment — stand up from the ground (experimental) |
| `train.py` | PPO training script with CheckpointCallback and RecordCallback |
| `evaluate.py` | Headless evaluation — success rate, height, stillness, drift |
| `watch.py` | Live MuJoCo viewer for trained policy |
| `record_callback.py` | SB3 callback — renders MP4 snapshots during training |
| `requirements.txt` | Pinned dependencies |
| `NOTES.md` | Full technical documentation — reward formulas, hyperparameters, all 6 iterations |
| `BEWERBUNG.md` | German portfolio document (application material) |
| `praesentation.html` | Interactive HTML showcase — open locally in any browser |
| `praesentation_assets/` | Comparison images and video (iteration 1 vs 4 vs 6) |
| `insights/` | 5 domain-knowledge reports used during reward design |

---

## The journey — 6 iterations

| Iter | Steps | Success | Mean episode length | Key change |
|------|-------|---------|---------------------|------------|
| 1 | 3M | 0% | 159 steps | Height + uprightness only → wild oscillation, no convergence |
| 2 | 5M | 5% | 456 steps | **Breakthrough:** pose-tracking (19 joints → `home` keyframe) added |
| 3 | 15M | 25% | 594 steps | Simple resume; stable but slow gradient |
| 4 | 25M | 83% | 930 steps | Learning rate 3e-4 → 1e-4 after clip_fraction=0.76 signal |
| 5 | 35M | 90% | 949 steps | Stillness penalties: XY drift + root velocity → robot stopped wandering |
| 6 | 45M | **97%** | **990 steps** | **Anatomical differentiation:** lower body (legs, ×0.020) vs. upper (arms, ×0.0005) + foot-lock |

**Final metrics (Phase 1):**

| Metric | Value | Target |
|--------|-------|--------|
| Success rate | 97% | ≥ 80% |
| Mean episode length | 990 / 1000 steps | — |
| Lower-body joint velocity | 0.23 rad/s | < 0.30 ✅ |
| Foot position drift | < 0.05 m over 10 s | — ✅ |
| Ankle actuator load | 82% of ctrlrange | — |

---

## Key insight: Human-in-the-Loop reward engineering

This project is not "run PPO and wait." The core workflow is iterative reward design
driven by human domain heuristics.

**Iteration 1→2:** After seeing wild oscillation, the fix came from biomechanics literature:
pose-reference rewards work for humanoid standing. Importing that prior knowledge (not
re-deriving it from scratch) made iteration 2 converge.

**Iteration 5→6:** The robot stood 90% of the time but drifted 3.4 m horizontally.
The naive fix is to penalize all joint velocities equally.
The biomechanically correct fix: legs must be still (support base), arms may swing (angular
momentum compensation). Result: 40:1 lower/upper velocity penalty ratio → 97% success,
no drift, arms actively used for balance.

---

## Domain-insight reports

The `insights/` folder contains 5 structured reports written before/during reward design:

| File | Topic |
|------|-------|
| `01_stillstand.md` | Stillness heuristics: lower/upper differentiation, foot lock |
| `02_aufstehen.md` | Phase 2 stand-up design: height-gain reward, longer horizon |
| `03_aktor_einstellungen.md` | Actuator analysis: ankle 82% load, torque mapping |
| `04_lerncurriculum.md` | 5-phase curriculum inspired by infant motor development |
| `05_aufstehen_choreografie.md` | Stand-up choreography: COM shift strategy |

---

## What's not done yet

- **Sim-to-real transfer** — everything runs in MuJoCo simulation. No real H1 hardware tested.
- **Perfect stillness** — 0.23 rad/s lower-body velocity; target for "statue" quality would be < 0.10 rad/s. Ankle actuators at 82% load are the bottleneck.
- **Phase 2 (stand up from ground)** — `h1_standup_env.py` exists and is designed, but training with sufficient budget is not complete.
- **ROS2 integration** — uses MuJoCo API + Gymnasium directly. No ROS2 bridge, no real-time constraints.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Author

**Ai-Crafters** ([ai-crafters.io](https://ai-crafters.io))
📧 info@ai-crafters.io
