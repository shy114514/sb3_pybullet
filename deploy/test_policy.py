#!/usr/bin/env python3
"""Run an exported TorchScript policy in a PyBullet backend environment."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs import get_available_backends, make_env
from utils import get_env_cfg, load_yaml_config

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Test an exported TorchScript push-task policy.")
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to the exported .pt policy.",
    )
    parser.add_argument(
        "--backend",
        default="pybullet",
        choices=get_available_backends(),
        help="Environment backend.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array", "none"],
        help="Environment render mode.",
    )
    parser.add_argument(
        "--obs-type",
        default="state",
        choices=["state"],
        help="Observation type.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for policy and world-model inference.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=8,
        help="Environment difficulty set before each episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    return parser.parse_args()


def policy_action(policy, obs: np.ndarray, device: str) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy(obs_tensor)
    return action.squeeze(0).cpu().numpy()


def run_episode(env, policy, episode: int, args):
    obs, _ = env.reset(seed=args.seed + episode)
    env.unwrapped.set_difficulty(args.difficulty)

    episode_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info = {}

    while not terminated and not truncated:
        action = policy_action(policy, obs, args.device)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

    return {
        "reward": episode_reward,
        "steps": steps,
        "success": bool(info["is_success"]),
    }


def main():
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    cfg = get_env_cfg(load_yaml_config(args.config))

    policy = torch.jit.load(args.policy, map_location=args.device)
    policy.eval()

    env = make_env(
        backend=args.backend,
        cfg=cfg,
        render_mode=render_mode,
        obs_type=args.obs_type,
        device=args.device,
    )

    print(f"Policy: {os.path.abspath(args.policy)}")
    print(f"Backend: {args.backend}")
    print(f"Episodes: {args.episodes}")

    results = []
    for episode in range(args.episodes):
        result = run_episode(env, policy, episode, args)
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAILED"
        print(
            f"Episode {episode + 1}: {status} | "
            f"reward={result['reward']:.2f} | steps={result['steps']}"
        )

    env.close()

    rewards = np.array([result["reward"] for result in results], dtype=np.float32)
    steps = np.array([result["steps"] for result in results], dtype=np.float32)
    success_rate = np.mean([result["success"] for result in results]) * 100.0

    print(
        f"Summary: success_rate={success_rate:.1f}% | "
        f"avg_reward={rewards.mean():.2f} | avg_steps={steps.mean():.1f}"
    )


if __name__ == "__main__":
    main()
