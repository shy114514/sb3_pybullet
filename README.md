# rlBullet

PyBullet-only reinforcement learning project for the push task from
`pybullet_PINN`. It keeps two environment versions:

- `pybullet`: PyBullet physics-engine transition.
- `pybullet_wm`: PyBullet scene reset/observation/reward with PINN world-model
  transition.

Isaac Lab and other simulator backends are intentionally not included.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --backend pybullet --timesteps 1000000
python train.py --backend pybullet_wm --timesteps 1000000 --device cpu
```

Common options:

```bash
python train.py --backend pybullet --n-envs 12 --save-freq 100000
python train.py --backend pybullet --load-checkpoint ./runs/20260607_120000_pybullet/
python train.py --backend pybullet --config configs/config.yaml
```

Use `--eval-freq 0` to disable periodic evaluation during training. Use
`--test` with `--n-envs 1` for interactive action input.

Training records, including TensorBoard logs, copied configs, and periodic
evaluation outputs, are written under `runs/<timestamp>_<backend>/`. Model files
and checkpoints are stored in that run's `models/` subdirectory.

## Evaluate

```bash
python evaluate.py --backend pybullet --model-path ./runs/run/models/ppo_push_robot.zip
python evaluate.py --backend pybullet_wm --model-path ./runs/run/models/ppo_push_robot.zip --device cpu
python evaluate.py --backend pybullet --model-path ./runs/run/models/checkpoints/ --save-video
```

Evaluation writes plots and optional videos under the model directory's
`results/` folder.

## Project Layout

```text
rlBullet/
├── configs/config.yaml
├── envs/
│   ├── assets/
│   ├── WM_models/
│   ├── base/
│   ├── pybullet/
│   └── factory.py
├── evaluate.py
├── requirements.txt
└── train.py
```

## World-Model Weights

The WM backend loads these files by default:

- `envs/WM_models/v_prediction.pth`
- `envs/WM_models/w_prediction.pth`

Override them with `WM_V_MODEL_PATH` and `WM_W_MODEL_PATH` when needed.
