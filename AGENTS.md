# Repository Guidelines

## Project Structure & Module Organization

This is a Python reinforcement learning project for PyBullet push-task environments. Top-level entry points are `train.py` for PPO training and `evaluate.py` for model evaluation. Shared environment creation lives in `envs/factory.py`. Environment implementations are under `envs/pybullet/`, with reusable configuration and base logic in `envs/base/`. Simulation assets are stored in `envs/assets/`, and world-model weights are stored in `envs/WM_models/`. Default training configuration is `configs/config.yaml`. Generated run outputs belong under `runs/`, with model files and checkpoints in each run's `models/` subdirectory.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, descriptive snake_case names for functions and variables, and PascalCase for classes such as environment classes and callbacks. Keep modules focused on one responsibility: environment logic in `envs/`, orchestration in entry-point scripts, and configuration in YAML. Prefer clear control flow over broad defensive checks; add validation only for realistic failure cases.
