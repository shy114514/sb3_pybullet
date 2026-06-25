# Policy Deployment

## Export

Export a Stable-Baselines3 PPO checkpoint to a TorchScript policy:

```bash
python deploy/export_policy.py \
  --checkpoint runs/<run>/models/ppo_push_robot.zip
```

For checkpoints, pass the checkpoint `.zip` file:

```bash
python deploy/export_policy.py \
  --checkpoint runs/<run>/models/checkpoints/ppo_push_robot_1000000_steps.zip
```

The exporter looks for the matching `VecNormalize` `.pkl` next to the checkpoint.
If needed, pass it explicitly:

```bash
python deploy/export_policy.py \
  --checkpoint runs/<run>/models/checkpoints/ppo_push_robot_1000000_steps.zip \
  --vecnormalize runs/<run>/models/checkpoints/ppo_push_robot_1000000_steps.pkl
```

Load the exported policy on GPU during deployment:

```python
import torch

policy = torch.jit.load("ppo_push_robot_1000000_steps_policy.pt", map_location="cuda")
policy.eval()

obs = torch.as_tensor(raw_obs, device="cuda", dtype=torch.float32)
with torch.no_grad():
    action = policy(obs)
```

The exported policy includes observation normalization from `VecNormalize`, so
deployment code should pass the raw observation vector.

## Policy Interface

The exported `.pt` file is a TorchScript module that runs the PPO actor in
deterministic mode.

Input:

- Raw environment observation, before `VecNormalize`.
- Shape: `(13,)` for a single observation, or `(N, 13)` for a batch.
- Type: `torch.float32`.
- Device: same device used when loading the policy.

Output:

- Continuous action in the environment action space.
- Shape: `(1, 2)` for one observation, or `(N, 2)` for a batch.
- Type: `torch.float32`.
- Range: PPO deterministic actor output. The environment action space is
  `Box(-1, 1, shape=(2,))`; the current environment clips actions in `step()`.
- Meaning: 2D end-effector delta command in the object heading frame.

Minimal usage:

```python
import torch

device = "cuda"
policy = torch.jit.load("ppo_push_robot_policy.pt", map_location=device)
policy.eval()

raw_obs = env.reset()[0]
obs = torch.as_tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)

with torch.no_grad():
    action = policy(obs)

action_np = action.squeeze(0).cpu().numpy()
obs, reward, terminated, truncated, info = env.step(action_np)
```

The policy already applies the observation mean, variance, clipping, and
epsilon from the exported `VecNormalize` statistics. Do not normalize the
observation again before passing it to the policy.

## Test

Run an exported `.pt` policy in the PyBullet environment:

```bash
python deploy/test_policy.py \
  --policy runs/<run>/models/ppo_push_robot_policy.pt \
  --backend pybullet \
  --episodes 5
```

Run the same policy against the PyBullet world-model backend:

```bash
python deploy/test_policy.py \
  --policy runs/<run>/models/ppo_push_robot_policy.pt \
  --backend pybullet_wm \
  --device cuda
```

Useful options:

```bash
python deploy/test_policy.py \
  --policy runs/<run>/models/ppo_push_robot_policy.pt \
  --backend pybullet \
  --render-mode none \
  --difficulty 8 \
  --seed 42
```

The test script prints per-episode success, reward, step count, and a final
summary with success rate, average reward, and average steps.
