"""
Base environment interface for push task.

This module defines the abstract interface shared by the PyBullet physics
environment and the PyBullet world-model environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class BasePushEnv(ABC):
    """Abstract base class for push task environment.

    All environments implement this interface to stay compatible with the
    training and evaluation scripts.

    The environment implements a robot arm pushing task:
    - Robot must push a rectangular object to a target position and orientation
    - Observation space: 19D state vector
    - Action space: 2D continuous end-effector delta (dx, dy)

    Observation features (state mode, 19D):
        1. ee_xy (2): End-effector position
        2. obj_xy (2): Object position
        3. target_xy (2): Target position
        4. ee_to_obj_xy (2): Relative position (EE to object)
        5. obj_to_target_xy (2): Relative position (object to target)
        6. obj_vel_xy (2): Object linear velocity
        7. obj_yaw_sincos (2): Object orientation (sin, cos encoding)
        8. target_yaw_sincos (2): Target orientation (sin, cos encoding)
        9. yaw_error_sincos (2): Orientation error (sin, cos encoding)
        10. obj_angular_vel (1): Object angular velocity (Z-axis)
    """

    # Class attributes that subclasses must define
    BACKEND_NAME: str = "base"
    SUPPORTS_GPU_PARALLEL: bool = False

    def __init__(
        self,
        cfg: Any,
        render_mode: Optional[str] = None,
        obs_type: str = "state",
        num_envs: int = 1,
        device: str = "cpu",
    ):
        """Initialize the push environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode ("human", "rgb_array", or None).
            obs_type: Observation type. Only "state" is supported.
            num_envs: Number of parallel environments requested by wrappers.
            device: Device to use ("cpu" or "cuda").
        """
        self.cfg = cfg
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.num_envs = num_envs
        self.device = device

    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.

        Returns:
            Tuple of (observation, info_dict).
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Action to execute (2D: dx, dy).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up environment resources."""
        pass

    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        pass

    # Properties that subclasses must implement
    @property
    @abstractmethod
    def observation_space(self):
        """Return the observation space."""
        pass

    @property
    @abstractmethod
    def action_space(self):
        """Return the action space."""
        pass

    # Optional methods for debugging/visualization
    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position (for debugging)."""
        raise NotImplementedError

    def get_object_pose(self) -> Tuple[np.ndarray, float]:
        """Get current object position and yaw angle (for debugging)."""
        raise NotImplementedError

    def get_target_pose(self) -> Tuple[np.ndarray, float]:
        """Get target position and yaw angle (for debugging)."""
        raise NotImplementedError
