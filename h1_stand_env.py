"""Gymnasium environment: Unitree H1 humanoid learns to stand (keep balance)."""

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
FRAME_SKIP = 5            # control timestep ~= 5 * physics dt (~0.01s)
MAX_EPISODE_STEPS = 1000  # truncation horizon
RESET_NOISE = 0.01        # gaussian noise (std) on qpos/qvel at reset

# --- Reward weights (Iteration 2: pose-reference reward) ---
POSE_WEIGHT = 2.0          # DOMINANT: reward for holding the 'home' joint pose
ALIVE_BONUS = 0.5          # bonus per surviving step
UPRIGHT_WEIGHT = 0.3       # support term: torso height near target
ORIENTATION_WEIGHT = 0.3   # support term: torso staying vertical
CONTROL_COST_WEIGHT = 0.001        # penalty on torque effort

# --- Iteration 6: anatomically differentiated reward + foot lock ---
# Lower body (legs/hips/ankles) -- 10 joints, must stay still.
LOWER_JOINT_IDX = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
LOWER_JOINT_VEL_WEIGHT = 0.020     # strict: legs are the support base
# Upper body (torso/shoulders/arms) -- 9 joints, allowed to swing for balance.
UPPER_JOINT_IDX = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])
UPPER_JOINT_VEL_WEIGHT = 0.0005    # mild: arms are active balancers
FOOT_POS_LOCK_WEIGHT = 2.0         # penalize feet drifting from reset xy
FOOT_AIR_PENALTY = 0.5             # per foot that has no ground contact
ROOT_LIN_VEL_WEIGHT = 0.3          # kept from Iter 5: horizontal root velocity

TARGET_HEIGHT = 0.98       # torso height of the 'home' keyframe
HEIGHT_SHARPNESS = 12.0    # exp(-k * (z - target)^2)
POSE_SHARPNESS = 6.0       # exp(-k * mean((q - q_home)^2))

# --- Termination ---
FALL_HEIGHT = 0.5          # torso below this -> fallen over


class H1StandEnv(gymnasium.Env):
    """H1 humanoid standing-balance task driven by 19 torque motors."""

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
        self._left_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link"
        )
        self._right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link"
        )

        # Reference pose: 19 joint angles of the 'home' keyframe (qpos[7:26]).
        self._home_joint_qpos = self.model.key("home").qpos[7:26].copy()

        # Foot xy lock target; set per-episode in reset().
        self._foot_xy_init = np.zeros((2, 2))

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

        self._step_count = 0
        self._renderer = None

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

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        self.data.qpos += self.np_random.normal(
            0.0, RESET_NOISE, size=self.model.nq
        )
        self.data.qvel += self.np_random.normal(
            0.0, RESET_NOISE, size=self.model.nv
        )
        mujoco.mj_forward(self.model, self.data)

        # Lock target = actual foot xy at episode start (after noise).
        self._foot_xy_init[0] = self.data.xpos[self._left_foot_id, :2]
        self._foot_xy_init[1] = self.data.xpos[self._right_foot_id, :2]

        self._step_count = 0
        return self._get_obs(), {}

    def _feet_on_ground(self):
        """Return (left_contact, right_contact) using MuJoCo contact list."""
        left_id, right_id = self._left_foot_id, self._right_foot_id
        left_c = right_c = False
        geom_body = self.model.geom_bodyid
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = geom_body[c.geom1], geom_body[c.geom2]
            if left_id in (b1, b2):
                left_c = True
            if right_id in (b1, b2):
                right_c = True
        return left_c, right_c

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
        control_cost = CONTROL_COST_WEIGHT * float(
            np.sum(np.square(action))
        )
        # Stillness penalties (Iteration 6): differentiated joint velocity.
        joint_qvel = self.data.qvel[6:25]
        lower_vel_cost = LOWER_JOINT_VEL_WEIGHT * float(
            np.sum(np.square(joint_qvel[LOWER_JOINT_IDX]))
        )
        upper_vel_cost = UPPER_JOINT_VEL_WEIGHT * float(
            np.sum(np.square(joint_qvel[UPPER_JOINT_IDX]))
        )

        # Foot position lock: drift of each foot from its reset xy.
        left_foot_drift = float(np.linalg.norm(
            self.data.xpos[self._left_foot_id, :2] - self._foot_xy_init[0]
        ))
        right_foot_drift = float(np.linalg.norm(
            self.data.xpos[self._right_foot_id, :2] - self._foot_xy_init[1]
        ))
        foot_lock_cost = FOOT_POS_LOCK_WEIGHT * (left_foot_drift + right_foot_drift)

        # Foot-air penalty: each foot losing ground contact costs.
        left_c, right_c = self._feet_on_ground()
        foot_air_left = 0 if left_c else 1
        foot_air_right = 0 if right_c else 1
        foot_air_cost = FOOT_AIR_PENALTY * (foot_air_left + foot_air_right)

        # Root linear velocity (xy) -- kept from Iter 5.
        root_lin_speed = float(np.linalg.norm(self.data.qvel[0:2]))
        root_lin_vel_cost = ROOT_LIN_VEL_WEIGHT * root_lin_speed

        reward = (
            pose + alive + upright + orientation
            - control_cost
            - lower_vel_cost - upper_vel_cost
            - foot_lock_cost - foot_air_cost
            - root_lin_vel_cost
        )

        terminated = height < FALL_HEIGHT
        truncated = self._step_count >= MAX_EPISODE_STEPS

        info = {
            "height": height,
            "uprightness": uprightness,
            "reward_pose": pose,
            "reward_upright": upright,
            "reward_orientation": orientation,
            "control_cost": control_cost,
            "lower_vel_cost": lower_vel_cost,
            "upper_vel_cost": upper_vel_cost,
            "foot_lock_cost": foot_lock_cost,
            "foot_air_cost": foot_air_cost,
            "foot_air_left": foot_air_left,
            "foot_air_right": foot_air_right,
            "root_lin_vel_cost": root_lin_vel_cost,
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
