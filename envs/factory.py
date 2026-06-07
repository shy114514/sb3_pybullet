"""Environment factory for PyBullet push-task environments."""

from typing import Any, List, Optional
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from .base import BasePushEnvConfig
from .pybullet import PyBulletPushEnv, WMPyBulletPushEnv


def get_available_backends() -> List[str]:
    """Get the PyBullet backends supported by this project."""
    return ["pybullet", "pybullet_wm"]


def make_env(
    backend: str = "pybullet",
    cfg: Optional[BasePushEnvConfig] = None,
    render_mode: Optional[str] = None,
    obs_type: str = "state",
    num_envs: int = 1,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """Create a single push-task environment.

    Args:
        backend: Backend to use ("pybullet" or "pybullet_wm").
        cfg: Environment configuration. If None, uses default config.
        render_mode: Render mode ("human", "rgb_array", or None).
        obs_type: Observation type ("state" or "image").
        num_envs: Accepted for API compatibility; PyBullet creates one env here.
        device: Device for world-model inference ("cpu" or "cuda").
        max_episode_steps: Override max episode steps from config.

    Returns:
        Gymnasium-compatible environment.

    Raises:
        ValueError: If backend is not available.
    """
    available = get_available_backends()
    if backend not in available:
        raise ValueError(
            f"Backend '{backend}' is not available. "
            f"Available backends: {available}"
        )

    if cfg is None:
        cfg = BasePushEnvConfig()

    if max_episode_steps is not None:
        cfg.max_episode_steps = max_episode_steps

    if backend == "pybullet":
        env = PyBulletPushEnv(
            cfg=cfg,
            render_mode=render_mode,
            obs_type=obs_type,
            num_envs=1,
            device=device,
        )
    elif backend == "pybullet_wm":
        env = WMPyBulletPushEnv(
            cfg=cfg,
            render_mode=render_mode,
            obs_type=obs_type,
            num_envs=1,
            device=device,
        )

    return TimeLimit(env, max_episode_steps=cfg.max_episode_steps)


def make_vec_env(
    backend: str = "pybullet",
    cfg: Optional[BasePushEnvConfig] = None,
    n_envs: int = 12,
    obs_type: str = "state",
    device: str = "cpu",
    vec_env_cls: Optional[type] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    """Create vectorized PyBullet push-task environments.

    Args:
        backend: Backend to use ("pybullet" or "pybullet_wm").
        cfg: Environment configuration. If None, uses default config.
        n_envs: Number of parallel environments.
        obs_type: Observation type ("state" or "image").
        device: Device for world-model inference ("cpu" or "cuda").
        vec_env_cls: VecEnv class to use (default: SubprocVecEnv).
        max_episode_steps: Override max episode steps from config.
        render_mode: Render mode for each environment.

    Returns:
        Vectorized environment compatible with Stable Baselines 3.
    """
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    if cfg is None:
        cfg = BasePushEnvConfig()

    if max_episode_steps is not None:
        cfg.max_episode_steps = max_episode_steps

    env_cls_by_backend = {
        "pybullet": PyBulletPushEnv,
        "pybullet_wm": WMPyBulletPushEnv,
    }
    if backend not in env_cls_by_backend:
        raise ValueError(f"Unknown backend: {backend}")

    if backend == "pybullet":
        device = "cpu"
    if vec_env_cls is None:
        vec_env_cls = SubprocVecEnv

    env_cls = env_cls_by_backend[backend]

    def make_env_fn():
        env = env_cls(
            cfg=cfg,
            render_mode=render_mode,
            obs_type=obs_type,
            device=device,
        )
        return TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    env = sb3_make_vec_env(
        make_env_fn,
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
        monitor_kwargs={"info_keywords": ("is_success",)},
    )

    if obs_type == "state":
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

    return env


def load_config_from_yaml(yaml_path: str) -> BasePushEnvConfig:
    """Load environment configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        BasePushEnvConfig loaded from the file.
    """
    import yaml

    with open(yaml_path, 'r', encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return BasePushEnvConfig.from_dict(config_dict)
