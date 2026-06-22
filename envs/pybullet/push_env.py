"""
PyBullet implementation of the push task environment.

This module implements the push task using PyBullet physics engine.
It uses state-based observations.
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import os
import time
from typing import Any, Dict, Optional, Tuple
from collections import namedtuple

from ..base import BasePushEnv


class PyBulletPushEnv(BasePushEnv, gym.Env):
    """PyBullet implementation of push task environment.

    Uses Kuka iiwa 7-DOF robot arm to push a rectangular object
    to a target position and orientation.

    Physics: 120Hz simulation, 10Hz control (12 physics steps per action)
    """

    BACKEND_NAME = "pybullet"
    SUPPORTS_GPU_PARALLEL = False

    def __init__(
        self,
        cfg: Any,
        render_mode: Optional[str] = None,
        obs_type: str = "state",
        num_envs: int = 1,
        device: str = "cpu",
    ):
        """Initialize PyBullet push environment.

        Args:
            cfg: Environment configuration.
            render_mode: "human" for GUI, "rgb_array" for rendering, None for headless.
            obs_type: Only "state" is supported.
            num_envs: Ignored for PyBullet (always 1).
            device: Ignored for PyBullet (always CPU).
        """
        super().__init__(cfg, render_mode, obs_type, num_envs=1, device=device)

        if self.obs_type != "state":
            raise ValueError("PyBulletPushEnv only supports state observations.")

        self.difficulty = 0
        self.distance_threshold = self.cfg.success_threshold
        self.orientation_threshold = self.cfg.orientation_threshold
        self.asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
        self.fixed_ee_z = self.cfg.fixed_ee_z
        # PushT-style action and reset settings.
        self.step_size = self.cfg.step_size
        self.control_freq_inv = self.cfg.control_freq_inv
        self.obj_x_range = self.cfg.obj_x_range
        self.obj_y_range = self.cfg.obj_y_range
        self.target_x_range = self.cfg.target_x_range
        self.target_y_range = self.cfg.target_y_range
        self.t_block_base_z = 0.014

        # Define spaces
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )


        # Connect to PyBullet
        self.phisics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        self.dt = 1.0 / 240.0  # Internal physics step
        p.setTimeStep(self.dt)
        self.smootherstep_alphas = self._make_smootherstep_alphas(self.control_freq_inv)

        self.step_count = 0

        # Target pose (set in reset)
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.target_orn = np.array([0.0, 0.0, 0.0, 1.0])

        # Reward state for the PushT-style task.
        self.alpha = self._cfg_value("alpha", "alpha", 5.0)
        self.w_pos = self._cfg_value("w_pos", "position_progress_coef", 100.0)
        self.w_pose = self._cfg_value("w_pose", "orientation_progress_coef", 100.0)
        self.w_align = self._cfg_value("w_align", "alignment_coef", 2.0)
        self.gamma = self._cfg_value("gamma", "gamma", 0.001)
        self.success_bonus = self._cfg_value("successBonus", "success_bonus", 100.0)
        self.step_penalty = self._cfg_value("stepPenalty", "step_penalty", -0.1)
        self.crash_penalty = self._cfg_value("crashPenalty", "crash_penalty", -100.0)
        self.crash_delta_thresh = self._cfg_value("crashDeltaThresh", "crash_delta_thresh", 0.1)
        self.base_pos_thresh = self._cfg_value("posThresh", "success_threshold", 0.03)
        self.pos_thresh = self.base_pos_thresh
        self.pos_deadzone = self._cfg_value("posDeadzone", "pos_deadzone", 0.06)
        self.base_ori_thresh = self._cfg_value("oriThresh", "orientation_threshold", 1.0)
        self.ori_thresh = self.base_ori_thresh
        self.actions = np.zeros(2, dtype=np.float32)
        self.prev_action_world = np.zeros(2, dtype=np.float32)
        self.prev_ee_to_obj_dist = 0.0
        self.prev_obj_to_tar_dist = 0.0
        self.prev_ori_err = 0.0
        self.prev_alignment_score = 0.0
        self.prev_obj_pos = np.zeros(2, dtype=np.float32)
        self.prev_obj_heading = 0.0
        self.task_reward = 0.0
        self.success_buf = False

        # Fixed end-effector orientation (pointing down)
        self.fixed_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])

        # Camera matrices for rendering and video recording.
        self._load_static_resources()
        self._create_dynamic_actors()
        self._setup_cameras()

    def _cfg_value(self, primary_name: str, fallback_name: str, default):
        if hasattr(self.cfg, primary_name):
            return getattr(self.cfg, primary_name)
        if hasattr(self.cfg, fallback_name):
            return getattr(self.cfg, fallback_name)
        return default

    def _make_smootherstep_alphas(self, steps: int) -> np.ndarray:
        t = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
        alphas = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
        return np.diff(alphas, prepend=np.float32(0.0)).astype(np.float32)

    def _load_static_resources(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        
        # # Robot
        # p.setAdditionalSearchPath(self.asset_dir)
        # base_pos = (0, 0, 0)
        # base_ori = p.getQuaternionFromEuler((0, 0, 0))
        # # Ensure file exists or handle error
        # try:
        #     self.robotId = p.loadURDF("urdf/ur5_robotiq_85.urdf", base_pos, base_ori, useFixedBase=True)
        # except Exception as e:
        #     print(f"Error loading robot URDF from {self.asset_dir}: {e}")
        #     raise e

        # self.eef_id = 7
        # self.fixed_orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
        
        # # Joint setup
        # self.arm_num_dofs = 6
        # self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        # self.__parse_joint_info__()
        # self.__setup_mimic_joints__()

    def draw_local_axes(self, body_unique_id, link_index=-1, line_length=0.5, line_width=3):
        """
        在指定的刚体/链接上绘制局部坐标系的 X(红), Y(绿), Z(蓝) 轴。
        
        参数:
        body_unique_id: 模型的ID
        link_index: 链接索引 (-1 表示 Base)
        line_length: 绘制轴的长度
        """
        # 1. 获取当前位姿
        if link_index == -1:
            # 获取基座 (Base) 的位置和四元数
            pos, quat = p.getBasePositionAndOrientation(body_unique_id)
        else:
            # 获取特定链接 (Link) 的世界状态
            link_state = p.getLinkState(body_unique_id, link_index)
            pos = link_state[4] # worldLinkFramePosition
            quat = link_state[5] # worldLinkFrameOrientation

        # 2. 将四元数转换为 3x3 旋转矩阵
        # 旋转矩阵的列向量正是局部坐标系 XYZ 轴在世界坐标系下的方向向量
        rot_matrix = p.getMatrixFromQuaternion(quat)
        
        # 提取局部 X, Y, Z 轴的方向向量
        # rot_matrix 是一维数组，长度为9，按行优先排列：[R11, R12, R13, R21, R22, R23, R31, R32, R33]
        x_axis = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        y_axis = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
        z_axis = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

        # 3. 计算轴的终点坐标 (起点 pos + 方向向量 * 长度)
        x_end = [pos[i] + x_axis[i] * line_length for i in range(3)]
        y_end = [pos[i] + y_axis[i] * line_length for i in range(3)]
        z_end = [pos[i] + z_axis[i] * line_length for i in range(3)]

        # 4. 绘制线条 (RGB 分别对应 XYZ)
        # 返回的是 debug item ID，如果需要动态更新，可以在下一次绘制时传入 replaceItemUniqueId
        p.addUserDebugLine(pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=line_width) # 红 = X
        p.addUserDebugLine(pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=line_width) # 绿 = Y
        p.addUserDebugLine(pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=line_width) # 蓝 = Z

    def _create_dynamic_actors(self):
        
        self.ee_pos = np.array([0, 0, self.fixed_ee_z])
        radius = 0.005
        length = 0.05
        mass = 100

        ee_vis = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=[0.8, 0.2, 0.2, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0, 0, length/2]
        )
        ee_col = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=length,
            collisionFramePosition=[0, 0, length/2]
        )

        self.eeId = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=ee_col,
            baseVisualShapeIndex=ee_vis,
            basePosition=self.ee_pos,  # 初始位置
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]) # 默认 Z 轴朝上
        )
        p.changeDynamics(self.eeId, -1, lateralFriction=0.8)
        
        self.constraintId = p.createConstraint(
            parentBodyUniqueId=self.eeId,
            parentLinkIndex=-1,
            childBodyUniqueId=-1, # -1 代表世界坐标系
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self.ee_pos # 初始目标位置
        )

        p.setAdditionalSearchPath(self.asset_dir)
        t_block_mesh_path = os.path.join(self.asset_dir, "object.obj")
        object_col = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=t_block_mesh_path,
        )
        object_vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=t_block_mesh_path,
            rgbaColor=[0.8, 0.2, 0.2, 1.0],
        )
        self.objectId = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=object_col,
            baseVisualShapeIndex=object_vis,
            basePosition=[0, 0, self.t_block_base_z],
            baseOrientation=self._t_block_orientation(0.0),
        )
        # self.objectId = p.loadURDF(
        #     "t_block.urdf",
        #     basePosition=[0, 0, self.t_block_base_z],
        #     baseOrientation=self._t_block_orientation(0.0),
        #     flags=p.URDF_USE_INERTIA_FROM_FILE,
        #     useFixedBase=False,
        # )
        p.changeDynamics(self.objectId, -1, lateralFriction=0.6, spinningFriction=0.01)

        target_vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=t_block_mesh_path,
            rgbaColor=[0.0, 1.0, 0.0, 0.5],
        )
        self.targetId = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_vis,
            basePosition=[0, 0, self.t_block_base_z],
            baseOrientation=self._t_block_orientation(0.0),
        )
        # self.targetId = p.loadURDF(
        #     "t_block_vision.urdf",
        #     basePosition=[0, 0, self.t_block_base_z],
        #     baseOrientation=self._t_block_orientation(0.0),
        #     flags=p.URDF_USE_INERTIA_FROM_FILE,
        #     useFixedBase=True,
        # )

        # # --- 尺寸定义 (与原代码一致) ---
        # base_half_extents = [0.1, 0.025, 0.025]  # 横杠
        # link_half_extents = [0.025, 0.075, 0.025] # 竖杠
        # link_pos = [0, -0.1, 0] 

        # # ==================================
        # # 1. 创建可操作物体 (红色 T 形块)
        # # ==================================
        # baseCol = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half_extents)
        # baseVis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=[0.8, 0.1, 0.1, 1])
        
        # linkCol = p.createCollisionShape(p.GEOM_BOX, halfExtents=link_half_extents)
        # linkVis = p.createVisualShape(p.GEOM_BOX, halfExtents=link_half_extents, rgbaColor=[0.8, 0.1, 0.1, 1])
        
        # self.objectId = p.createMultiBody(
        #     baseMass=0.5,
        #     baseCollisionShapeIndex=baseCol,
        #     baseVisualShapeIndex=baseVis,
        #     basePosition=[0, 0, 0.025],
        #     linkMasses=[0.5],
        #     linkCollisionShapeIndices=[linkCol],
        #     linkVisualShapeIndices=[linkVis],
        #     linkPositions=[link_pos],
        #     linkOrientations=[[0, 0, 0, 1]],
        #     linkInertialFramePositions=[[0, 0, 0]],
        #     linkInertialFrameOrientations=[[0, 0, 0, 1]],
        #     linkParentIndices=[0],
        #     linkJointTypes=[p.JOINT_FIXED],
        #     linkJointAxis=[[0, 0, 0]]
        # )

        # p.changeDynamics(self.objectId, -1, lateralFriction=0.6, spinningFriction=0.1)
        # p.changeDynamics(self.objectId, 0, lateralFriction=0.6, spinningFriction=0.1)

        # # ==================================
        # # 2. 创建目标 (半透明绿色 T 形块 Ghost)
        # # ==================================
        # # 注意：Target 不需要 CollisionShape (设为 -1)，以免物理干扰
        # # 颜色设置为半透明绿色 (Alpha=0.3)
        # targetBaseVis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half_extents, rgbaColor=[0, 1, 0, 0.3])
        # targetLinkVis = p.createVisualShape(p.GEOM_BOX, halfExtents=link_half_extents, rgbaColor=[0, 1, 0, 0.3])

        # self.targetId = p.createMultiBody(
        #     baseMass=0, # 静态物体
        #     baseCollisionShapeIndex=-1, # 无碰撞
        #     baseVisualShapeIndex=targetBaseVis,
        #     basePosition=[0, 0, 0],
        #     # 必须构建完全相同的 Link 结构，才能在视觉上成为 T 形
        #     linkMasses=[0],
        #     linkCollisionShapeIndices=[-1], # 无碰撞
        #     linkVisualShapeIndices=[targetLinkVis],
        #     linkPositions=[link_pos],
        #     linkOrientations=[[0, 0, 0, 1]],
        #     linkInertialFramePositions=[[0, 0, 0]],
        #     linkInertialFrameOrientations=[[0, 0, 0, 1]],
        #     linkParentIndices=[0],
        #     linkJointTypes=[p.JOINT_FIXED],
        #     linkJointAxis=[[0, 0, 0]]
        # )

    def _setup_cameras(self):
        """Set up camera matrices for rendering."""
        self.global_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, -0.5, 0.8],
            cameraTargetPosition=[0.5, 0, 0.1],
            cameraUpVector=[0, 0, 1]
        )
        self.global_proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=4.0 / 3.0,
            nearVal=0.1,
            farVal=100.0
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.step_count = 0

        center_x, center_y = 0., 0.

        # --- 1. 重置物体 (Object) ---
        object_x = center_x + self.np_random.uniform(*self.obj_x_range)
        object_y = center_y + self.np_random.uniform(*self.obj_y_range)
        object_yaw = self.np_random.uniform(-np.pi, np.pi)
        
        obj_pos = [object_x, object_y, self.t_block_base_z]
        obj_orn = self._t_block_orientation(object_yaw)
        
        p.resetBasePositionAndOrientation(self.objectId, obj_pos, obj_orn)
        p.resetBaseVelocity(self.objectId, [0, 0, 0], [0, 0, 0])

        # --- 2. 重置目标 (Target) ---
        target_x = center_x + self.np_random.uniform(*self.target_x_range)
        target_y = center_y + self.np_random.uniform(*self.target_y_range)
        target_yaw = self.np_random.uniform(-np.pi, np.pi)

        self.target_pos = np.array([target_x, target_y, self.t_block_base_z])
        self.target_yaw = target_yaw
        target_orn = self._t_block_orientation(target_yaw)
        self.target_orn = np.array(target_orn, dtype=np.float32)
        
        p.resetBasePositionAndOrientation(self.targetId, self.target_pos, target_orn)

        # 步进几帧让物体落稳
        for _ in range(20): p.stepSimulation()
        
        # Init Reward Vars
        obj_pos_3d, obj_orn = p.getBasePositionAndOrientation(self.objectId)
        target_pos_3d, target_orn = p.getBasePositionAndOrientation(self.targetId)
        obj_pos = np.array(obj_pos_3d[:2], dtype=np.float32)
        target_pos = np.array(target_pos_3d[:2], dtype=np.float32)
        ee_pos = self.get_ee_position()[:2]

        self.actions[:] = 0.0
        self.prev_action_world[:] = 0.0
        self.task_reward = 0.0
        self.success_buf = False
        self.prev_obj_pos = obj_pos.copy()
        self.prev_obj_heading = self._object_heading_from_quat(obj_orn)
        self.prev_ee_to_obj_dist = np.linalg.norm(ee_pos - obj_pos)
        self.prev_obj_to_tar_dist = np.linalg.norm(target_pos - obj_pos)
        self.prev_ori_err = self._quat_orientation_error(obj_orn, target_orn)
        self.prev_alignment_score = self._alignment_score(obj_pos, ee_pos, target_pos)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.actions[:] = action

        _, obj_orn = p.getBasePositionAndOrientation(self.objectId)
        obj_heading = self._object_heading_from_quat(obj_orn)
        delta_pos = self._rotate_xy_obj_to_world(action * self.step_size, obj_heading)
        self.prev_action_world[:] = delta_pos
        target_ee_pos = self.ee_pos.copy()

        for alpha in self.smootherstep_alphas:
            target_ee_pos[:2] += alpha * delta_pos
            p.changeConstraint(self.constraintId, target_ee_pos)
            p.stepSimulation()
            # Enforce flat object constraint simply by re-asserting Z/Orientation IF necessary
            # But relying on correct friction/inertia is better physics.
            # If "teleport" is strictly needed, do it ONLY if object tips excessively.
            # Here we skip the hack to respect physics, assuming box CoM is low.

        if self.render_mode == "human":
            time.sleep(self.dt * self.control_freq_inv)

        self.step_count += 1
        
        # Reward first, then observation history update, matching PushT.
        reward, terminated, truncated = self._compute_reward()
        obs = self._get_obs()
        info = {}
        info['is_success'] = terminated

        return obs, reward, terminated, truncated, info
    
    def set_difficulty(self, difficulty: int):
        self.difficulty = difficulty
        self.pos_thresh = self.base_pos_thresh * (0.97 ** self.difficulty)
        self.ori_thresh = self.base_ori_thresh * (0.97 ** self.difficulty)
        self.distance_threshold = self.pos_thresh
        self.orientation_threshold = self.ori_thresh

    def _angle_normalize(self, angle):
        """将角度归一化到 [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _t_block_orientation(self, yaw: float):
        # object.obj uses local X as thickness; rotate it onto world Z, then apply planar yaw.
        flat_orn = p.getQuaternionFromEuler([0.0, math.pi / 2.0, 0.0])
        yaw_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        return p.multiplyTransforms([0, 0, 0], yaw_orn, [0, 0, 0], flat_orn)[1]

    def _quat_orientation_error(self, obj_quat, target_quat) -> float:
        dot_product = float(np.dot(np.asarray(obj_quat), np.asarray(target_quat)))
        abs_dot = np.clip(abs(dot_product), 0.0, 1.0 - 1e-6)
        return 2.0 * math.acos(abs_dot)

    def _alignment_score(self, obj_pos: np.ndarray, ee_pos: np.ndarray, target_pos: np.ndarray) -> float:
        vec_o_to_ee = ee_pos - obj_pos
        vec_t_to_o = obj_pos - target_pos
        ee_to_obj_dist = np.linalg.norm(vec_o_to_ee)
        obj_to_tar_dist = np.linalg.norm(vec_t_to_o)
        dir_t_to_o = vec_t_to_o / (obj_to_tar_dist + 1e-6)
        dir_o_to_ee = vec_o_to_ee / (ee_to_obj_dist + 1e-6)
        cos_sim = np.dot(dir_t_to_o, dir_o_to_ee)
        return (cos_sim + 1.0) / 2.0

    def _object_heading_from_quat(self, obj_quat) -> float:
        rot_matrix = p.getMatrixFromQuaternion(obj_quat)
        return math.atan2(rot_matrix[5], rot_matrix[2])

    def _rotate_xy_world_to_obj(self, vec_xy: np.ndarray, heading: float) -> np.ndarray:
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        return np.array([
            cos_h * vec_xy[0] + sin_h * vec_xy[1],
            -sin_h * vec_xy[0] + cos_h * vec_xy[1],
        ], dtype=np.float32)

    def _rotate_xy_obj_to_world(self, vec_xy: np.ndarray, heading: float) -> np.ndarray:
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        return np.array([
            cos_h * vec_xy[0] - sin_h * vec_xy[1],
            sin_h * vec_xy[0] + cos_h * vec_xy[1],
        ], dtype=np.float32)

    def _target_quat_in_object_frame(self, obj_quat, target_quat) -> np.ndarray:
        _, obj_inv_quat = p.invertTransform([0.0, 0.0, 0.0], obj_quat)
        _, target_obj_quat = p.multiplyTransforms(
            [0.0, 0.0, 0.0], obj_inv_quat, [0.0, 0.0, 0.0], target_quat
        )
        target_obj_quat = np.asarray(target_obj_quat, dtype=np.float32)
        if target_obj_quat[3] < 0.0:
            target_obj_quat *= -1.0
        return target_obj_quat

    def _update_observation_history(self, obj_pos: np.ndarray, obj_orn) -> None:
        self.prev_obj_pos = obj_pos.copy()
        self.prev_obj_heading = self._object_heading_from_quat(obj_orn)

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return self._get_state_obs()

    def _get_state_obs(self) -> np.ndarray:
        """Get 13D PushT state observation in the object heading frame."""
        self.ee_pos = self.get_ee_position()
        ee_pos = self.ee_pos[:2].astype(np.float32)
        obj_pos_3d, obj_orn = p.getBasePositionAndOrientation(self.objectId)
        target_pos_3d, target_orn = p.getBasePositionAndOrientation(self.targetId)
        obj_pos = np.array(obj_pos_3d[:2], dtype=np.float32)
        target_pos = np.array(target_pos_3d[:2], dtype=np.float32)
        obj_heading = self._object_heading_from_quat(obj_orn)

        ee_obj_pos = self._rotate_xy_world_to_obj(ee_pos - obj_pos, obj_heading)
        target_obj_pos = self._rotate_xy_world_to_obj(target_pos - obj_pos, obj_heading)
        target_obj_quat = self._target_quat_in_object_frame(obj_orn, target_orn)
        prev_action_obj = self._rotate_xy_world_to_obj(self.prev_action_world, obj_heading)
        obj_pos_delta = self._rotate_xy_world_to_obj(obj_pos - self.prev_obj_pos, obj_heading)
        obj_heading_delta = self._normalize_angle(obj_heading - self.prev_obj_heading)

        obs = np.concatenate((
            ee_obj_pos,
            target_obj_pos,
            target_obj_quat,
            prev_action_obj,
            obj_pos_delta,
            [obj_heading_delta],
        ))

        self._update_observation_history(obj_pos, obj_orn)

        return obs.astype(np.float32)

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        """Compute reward, terminated, truncated using the PushT-style formula."""
        obj_pos_3d, obj_orn = p.getBasePositionAndOrientation(self.objectId)
        target_pos_3d, target_orn = p.getBasePositionAndOrientation(self.targetId)
        obj_pos = np.array(obj_pos_3d[:2], dtype=np.float32)
        target_pos = np.array(target_pos_3d[:2], dtype=np.float32)
        ee_pos = self.get_ee_position()[:2]

        vec_o_to_ee = ee_pos - obj_pos
        vec_t_to_o = obj_pos - target_pos
        curr_ee_to_obj_dist = np.linalg.norm(vec_o_to_ee)
        curr_obj_to_tar_dist = np.linalg.norm(vec_t_to_o)
        curr_ori_err = self._quat_orientation_error(obj_orn, target_orn)

        dir_t_to_o = vec_t_to_o / (curr_obj_to_tar_dist + 1e-6)
        dir_o_to_ee = vec_o_to_ee / (curr_ee_to_obj_dist + 1e-6)
        cos_sim = np.dot(dir_t_to_o, dir_o_to_ee)
        curr_alignment_score = (cos_sim + 1.0) / 2.0
        alignment_decay = np.clip(curr_obj_to_tar_dist / (self.pos_thresh * 1.0) - 0.9, 0.0, 1.0)
        alignment_reward = self.w_align * (curr_alignment_score - self.prev_alignment_score) * alignment_decay

        approach_reward = self.alpha * (self.prev_ee_to_obj_dist - curr_ee_to_obj_dist)
        pos_reward = self.w_pos * (self.prev_obj_to_tar_dist - curr_obj_to_tar_dist)
        ori_reward = self.w_pose * (self.prev_ori_err - curr_ori_err)

        action_penalty = -self.gamma * np.sum(self.actions ** 2)
        step_penalty = self.step_penalty
        obj_planar_delta = np.linalg.norm(obj_pos - self.prev_obj_pos)
        crash_penalty = self.crash_penalty if obj_planar_delta > self.crash_delta_thresh else 0.0

        self.success_buf = self.success_buf or (
            curr_obj_to_tar_dist < self.pos_thresh and curr_ori_err < self.ori_thresh
        )
        success_bonus_term = float(self.success_buf) * self.success_bonus

        task_reward = approach_reward + pos_reward + ori_reward + alignment_reward
        reward = task_reward + action_penalty + step_penalty + crash_penalty + success_bonus_term
        self.task_reward += task_reward

        self.prev_ee_to_obj_dist = curr_ee_to_obj_dist
        self.prev_obj_to_tar_dist = curr_obj_to_tar_dist
        self.prev_ori_err = curr_ori_err
        self.prev_alignment_score = curr_alignment_score

        terminated = bool(self.success_buf)
        truncated = (not terminated) and (self.step_count >= self.cfg.max_episode_steps)

        return float(reward), terminated, truncated

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def reset_arm(self):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.robotId, joint_id, rest_pose)

    def move_gripper(self, open_length):
        # Simplified mimic logic trigger
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.robotId, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle)

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.robotId)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.robotId, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.robotId, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.robotId, self.mimic_parent_id,
                                   self.robotId, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            w, h, rgb, _, _ = p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=self.global_view_matrix,
                projectionMatrix=self.global_proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )
            rgb = np.array(rgb, dtype=np.uint8)
            rgb = np.reshape(rgb, (240, 320, 4))
            return rgb[:, :, :3]
        return None

    def close(self):
        """Close the environment."""
        p.disconnect()

    # Debug methods
    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        ee_state = p.getBasePositionAndOrientation(self.eeId)
        return np.array(ee_state[0])

    def get_object_pose(self) -> Tuple[np.ndarray, float]:
        """Get current object position and yaw angle."""
        pos, orn = p.getBasePositionAndOrientation(self.objectId)
        euler = p.getEulerFromQuaternion(orn)
        return np.array(pos), euler[2]

    def get_target_pose(self) -> Tuple[np.ndarray, float]:
        """Get target position and yaw angle."""
        return self.target_pos.copy(), self.target_yaw
