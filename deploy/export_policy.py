#!/usr/bin/env python3
"""Export a Stable-Baselines3 PPO checkpoint as a TorchScript policy."""

import argparse
import os
import pickle
from pathlib import Path

import torch
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser(description="Export PPO checkpoint policy to TorchScript.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the PPO checkpoint .zip file. The .zip suffix may be omitted.",
    )
    parser.add_argument(
        "--vecnormalize",
        default=None,
        help="Path to the VecNormalize .pkl file. If omitted, it is inferred from the checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .pt path. Defaults to <checkpoint_stem>_policy.pt next to the checkpoint, "
            "or in the parent directory when the checkpoint is inside a checkpoints folder."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.suffix != ".zip":
        path = path.with_suffix(path.suffix + ".zip") if path.suffix else Path(str(path) + ".zip")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


def find_vecnormalize_path(checkpoint_path: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"VecNormalize file not found: {path}")
        return path

    base_path = checkpoint_path.with_suffix("")
    candidates = [
        Path(str(base_path) + "_vecnormalize.pkl"),
        Path(str(base_path) + ".pkl"),
        checkpoint_path.parent / "vecnormalize.pkl",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "VecNormalize file not found. Pass --vecnormalize explicitly. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def default_output_path(checkpoint_path: Path) -> Path:
    output_name = f"{checkpoint_path.stem}_policy.pt"
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / output_name
    return checkpoint_path.with_name(output_name)


def load_vecnormalize_stats(vecnormalize_path: Path):
    with vecnormalize_path.open("rb") as f:
        vecnormalize = pickle.load(f)

    if not hasattr(vecnormalize, "obs_rms"):
        raise ValueError(f"{vecnormalize_path} does not contain VecNormalize observation stats.")

    mean = torch.as_tensor(vecnormalize.obs_rms.mean, dtype=torch.float32)
    var = torch.as_tensor(vecnormalize.obs_rms.var, dtype=torch.float32)
    clip_obs = float(vecnormalize.clip_obs)
    epsilon = float(vecnormalize.epsilon)

    return mean, var, clip_obs, epsilon


class NormalizedDeterministicPolicy(torch.nn.Module):
    """TorchScript wrapper that matches SB3 VecNormalize observation preprocessing."""

    def __init__(
        self,
        policy: torch.nn.Module,
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
        clip_obs: float,
        epsilon: float,
    ):
        super().__init__()
        self.policy = policy
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_var", obs_var)
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.obs_dim = obs_mean.numel()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(dtype=torch.float32).reshape(-1, self.obs_dim)
        normalized_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        normalized_obs = torch.clamp(normalized_obs, -self.clip_obs, self.clip_obs)
        return self.policy._predict(normalized_obs, deterministic=True)


def export_policy(checkpoint_path: Path, vecnormalize_path: Path, output_path: Path):
    model = PPO.load(checkpoint_path, device="cpu")
    model.policy.eval()

    obs_mean, obs_var, clip_obs, epsilon = load_vecnormalize_stats(vecnormalize_path)
    wrapper = NormalizedDeterministicPolicy(
        policy=model.policy,
        obs_mean=obs_mean,
        obs_var=obs_var,
        clip_obs=clip_obs,
        epsilon=epsilon,
    ).eval()

    example_obs = torch.zeros(1, obs_mean.numel(), dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_obs)
        traced.save(output_path)

        loaded = torch.jit.load(output_path, map_location="cpu")
        original_action = wrapper(example_obs)
        loaded_action = loaded(example_obs)

    if original_action.shape != loaded_action.shape:
        raise RuntimeError(
            "Export smoke test failed: "
            f"wrapper output shape {tuple(original_action.shape)} != "
            f"loaded output shape {tuple(loaded_action.shape)}"
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"VecNormalize: {vecnormalize_path}")
    print(f"Observation dim: {obs_mean.numel()}")
    print(f"Action shape: {tuple(loaded_action.shape)}")
    print(f"Exported policy: {output_path}")


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    vecnormalize_path = find_vecnormalize_path(checkpoint_path, args.vecnormalize)
    output_path = Path(args.output) if args.output else default_output_path(checkpoint_path)

    output_dir = output_path.parent
    if str(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_policy(checkpoint_path, vecnormalize_path, output_path)


if __name__ == "__main__":
    main()
