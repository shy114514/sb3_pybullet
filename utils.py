"""Configuration loading helpers for training and evaluation."""

import os
from datetime import datetime
from types import SimpleNamespace

import yaml


def load_yaml_config(config_path: str) -> dict:
    """Load the complete YAML configuration as a dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env_cfg(yaml_config: dict) -> SimpleNamespace:
    """Build the environment configuration from YAML environment sections."""
    env_values = {}
    for section_name in ("env", "reward"):
        env_values.update(yaml_config[section_name])

    return SimpleNamespace(**env_values)


def get_train_cfg(args, yaml_config: dict) -> dict:
    """Build the training and run configuration from YAML training settings."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join("./runs", f"{timestamp}_{args.backend}")
    model_dir = os.path.join(run_dir, "models")

    yaml_train_cfg = yaml_config.get("training", {})
    obs_train_cfg = yaml_train_cfg.get(args.obs_type, {})
    train_cfg = {
        "model_dir": model_dir,
        "run_dir": run_dir,
        "checkpoint_path": os.path.join(model_dir, "checkpoints"),
        "tensorboard_log": os.path.join(run_dir, "tensorboard"),
        "model_save_path": os.path.join(model_dir, "ppo_push_robot"),
        "model_name": "ppo_push_robot",
        "progress_update_freq": yaml_train_cfg.get("progress_update_freq", 32768),
        "curriculum_threshold": yaml_train_cfg.get("curriculum_threshold", 0.8),
        "policy_type": "MlpPolicy",
    }

    common_ppo_defaults = {
        "learning_rate": 0.0001,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    }
    ppo_kwargs = {
        key: obs_train_cfg.get(key, default)
        for key, default in common_ppo_defaults.items()
    }
    ppo_kwargs["policy_kwargs"] = dict(net_arch=dict(
        pi=obs_train_cfg.get("net_arch_pi", [256, 256, 128]),
        vf=obs_train_cfg.get("net_arch_vf", [256, 256, 128]),
    ))
    train_cfg["ppo_kwargs"] = ppo_kwargs

    return train_cfg
