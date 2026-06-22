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
python train.py --backend pybullet
python train.py --backend pybullet --timesteps 1000000 --n-envs 16
python train.py --backend pybullet_wm --timesteps 1000000 --device cpu
```

Common options:

```bash
python train.py --backend pybullet --config configs/config.yaml
python train.py --backend pybullet --run-dir ./runs/debug_pybullet
python train.py --backend pybullet --save-freq 100000
python train.py --backend pybullet --load-checkpoint ./runs/<timestamp>_pybullet/
python train.py --backend pybullet --eval-freq 0
python train.py --backend pybullet --no-eval-save-video
```

Training uses `pybullet` by default. The command-line default is
`--timesteps 1000000 --n-envs 16 --device cpu`. Periodic evaluation runs every
50,000 steps by default and writes videos/metrics under `periodic_eval/` inside
the run directory; use `--eval-freq 0` to disable it.

Training records, including TensorBoard logs, copied configs, and periodic
evaluation outputs, are written under `runs/<timestamp>_<backend>/`. Model files
and checkpoints are stored in that run's `models/` subdirectory.

## Environment Test

Run the interactive PyBullet environment test without training:

```bash
python train.py --backend pybullet --test --n-envs 1
python train.py --backend pybullet_wm --test --n-envs 1 --device cpu
```

Test mode opens the environment with human rendering, forces one environment,
and prompts for two action values such as `0.1 -0.2`. Press Enter to reuse the
previous action and type `q` to quit.

## Evaluate

```bash
python evaluate.py --backend pybullet
python evaluate.py --backend pybullet --model-path ./runs/<timestamp>_pybullet/models/ppo_push_robot.zip --episodes 20
python evaluate.py --backend pybullet_wm --model-path ./runs/<timestamp>_pybullet_wm/models/ppo_push_robot.zip --device cpu
python evaluate.py --backend pybullet --model-path ./runs/<timestamp>_pybullet/models/checkpoints/ --save-video
```

If `--model-path` is omitted, evaluation searches the latest run directory for
the selected backend. `--model-path` may point to a single `.zip` model, a
`models/` directory, a run directory, or a checkpoint directory. Evaluation
writes plots and optional videos under the model directory's `results/` folder.

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
