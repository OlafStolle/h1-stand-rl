"""Gymnasium environment: Unitree H1 humanoid learns to STAND UP from the ground."""

import os

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

# --- Paths ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "mujoco_menagerie", "unitree_h1", "scene.xml")
)

# --- Simulation ---
FRAME_SKIP = 5             # control timestep ~= 5 * physics dt (~0.01s)
MAX_EPISODE_STEPS = 1500   # longer horizon than Phase 1: getting up needs time
RESET_NOISE = 0.02         # gaussian noise (std) on qpos/qvel at reset
DOWNED_SETTLE_STEPS = 200  # passive fall steps to reach a "downed" pose

# --- Reward weights (Phase 2: stand up from the ground) ---
POSE_WEIGHT = 2.0          # DOMINANT: reward for reaching the 'home' joint pose
ALIVE_BONUS = 0.1          # low: must not pay for just lying still
UPRIGHT_WEIGHT = 0.5       # stronger than Phase 1: clear up-signal from downed
ORIENTATION_WEIGHT = 0.5   # stronger than Phase 1: same reason
HEIGHT_GAIN_WEIGHT = 5.0   # NEW: actively reward gaining torso height
CONTROL_COST_WEIGHT = 0.001        # penalty on torque effort
JOINT_VEL_COST_WEIGHT = 0.0001     # milder than Phase 1: standing up needs motion

TARGET_HEIGHT = 0.98       # torso height of the 'home' keyframe
HEIGHT_SHARPNESS = 12.0    # exp(-k * (z - target)^2)
POSE_SHARPNESS = 3.0       # milder than Phase 1: downed pose is far from home


class H1StandupEnv(gymnasium.Env):
    """H1 humanoid stand-up task driven by 19 torque motors. Starts on the ground."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        self._home_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )

        # Reference pose: 19 joint angles of the 'home' keyframe (qpos[7:26]).
        self._home_joint_qpos = self.model.key("home").qpos[7:26].copy()

        # Action: 19 normalized torques in [-1, 1], scaled to each motor's ctrlrange.
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        n_act = self.model.nu  # 19
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )

        # Observation: height(1) + root quat(4) + joint angles(19) + velocities(25) = 49
        obs_dim = 1 + 4 + (self.model.nq - 7) + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Determine "downed" pose once: drop from 'home' with zero control.
        self._downed_qpos, self._downed_qvel = self._compute_downed_pose()

        self._step_count = 0
        self._prev_height = 0.0
        self._renderer = None

    def _compute_downed_pose(self):
        """Let the robot fall naturally from 'home' with ctrl=0; capture rest state."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        self.data.ctrl[:] = 0.0
        for _ in range(DOWNED_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data, nstep=FRAME_SKIP)
        return self.data.qpos.copy(), self.data.qvel.copy()

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        obs = np.concatenate(
            [
                qpos[2:3],   # torso height (root z)
                qpos[3:7],   # root quaternion
                qpos[7:],    # 19 joint angles
                qvel[:],     # 25 velocities (root linear/angular + joints)
            ]
        )
        return obs.astype(np.float32)

    def _scale_action(self, action):
        """Map normalized action [-1, 1] to physical ctrlrange per motor."""
        action = np.clip(action, -1.0, 1.0)
        return self._ctrl_low + 0.5 * (action + 1.0) * (
            self._ctrl_high - self._ctrl_low
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Start from the precomputed downed pose, with small noise.
        self.data.qpos[:] = self._downed_qpos
        self.data.qvel[:] = self._downed_qvel
        self.data.qpos += self.np_random.normal(
            0.0, RESET_NOISE, size=self.model.nq
        )
        self.data.qvel += self.np_random.normal(
            0.0, RESET_NOISE, size=self.model.nv
        )
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_height = float(self.data.qpos[2])
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = self._scale_action(np.asarray(action))
        mujoco.mj_step(self.model, self.data, nstep=FRAME_SKIP)
        self._step_count += 1

        height = float(self.data.qpos[2])

        # Torso up-vector: third column of the torso rotation matrix.
        torso_zaxis = self.data.xmat[self._torso_id].reshape(3, 3)[:, 2]
        uprightness = float(torso_zaxis[2])  # projection on world Z

        # Dominant term: how close the 19 joints are to the 'home' pose.
        joint_qpos = self.data.qpos[7:26]
        pose_error = np.mean(
            np.square(joint_qpos - self._home_joint_qpos)
        )
        pose = POSE_WEIGHT * np.exp(-POSE_SHARPNESS * pose_error)

        alive = ALIVE_BONUS
        upright = UPRIGHT_WEIGHT * np.exp(
            -HEIGHT_SHARPNESS * (height - TARGET_HEIGHT) ** 2
        )
        orientation = ORIENTATION_WEIGHT * max(0.0, uprightness)
        # Reward only positive height gain (don't penalize natural settling).
        height_gain = HEIGHT_GAIN_WEIGHT * max(0.0, height - self._prev_height)
        control_cost = CONTROL_COST_WEIGHT * float(
            np.sum(np.square(action))
        )
        joint_vel_cost = JOINT_VEL_COST_WEIGHT * float(
            np.sum(np.square(self.data.qvel[6:25]))
        )
        reward = (
            pose + alive + upright + orientation + height_gain
            - control_cost - joint_vel_cost
        )

        self._prev_height = height

        # No termination on low torso height -- robot starts on the ground.
        terminated = False
        truncated = self._step_count >= MAX_EPISODE_STEPS

        info = {
            "height": height,
            "uprightness": uprightness,
            "reward_pose": pose,
            "reward_upright": upright,
            "reward_orientation": orientation,
            "reward_height_gain": height_gain,
            "control_cost": control_cost,
            "joint_vel_cost": joint_vel_cost,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
