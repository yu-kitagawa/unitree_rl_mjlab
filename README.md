# Unitree RL Mjlab

MuJoCo-based reinforcement learning and deployment for Unitree robots, built on
[mjlab](https://github.com/mujocolab/mjlab). This checkout includes G1 Velocity,
Navigation and OnlineMimic control, simulated camera/LiDAR sensors, and the
controller interfaces used by [TaskPromptRL](https://github.com/isri-aist/TaskPromptRL).

## Documentation

| Task | Guide |
|---|---|
| Install training dependencies and build G1 deploy | [Setup](doc/setup_en.md) · [中文](doc/setup_zh.md) |
| Start sim/real components individually with a gamepad | [Sim/real workflow](doc/sim_real_workflow.md) |
| Run the checked real setup and record Viser | [Real hardware quickstart](doc/real_hardware_quickstart.md) |
| Configure scenes and simulated sensors | [Simulator](simulate/README.md) |
| Install GLIM/Livox and use the shared settings | [GLIM setup](glim/README.md) |
| Recognize markers and calibrate the real camera | [ArUco](aruco/README.md) · [Camera calibration](doc/camera_calibration_audit.md) · [Sensor validation](doc/real_sensor_validation.md) |
| Understand root frames and timestamps | [Sensor frames](doc/sensor_frames.md) |
| Send goals and interpret arrival | [Navigation protocol](doc/navigation_goals.md) |
| Inspect controller paths, commands and measured posture | [Telemetry](doc/navigation_telemetry.md) · [Viewer protocol](doc/viewer_protocol.md) |
| Understand Mimic admission and reproduce a rejection | [Command continuity](doc/mimic_start_continuity.md) · [Offline replay](doc/mimic_admission_audit.md) |
| Run repeatable isolated simulations | [Simulation checks](doc/sim_check.md) · [Validation results](doc/validation_results.md) |
| Compare balance objectives | [Velocity / Mimic audit](doc/policy_balance_audit.md) |

## G1 box transport

TaskPromptRL generates reference motions and converts them in the sending Python
process using this repository's converter. `g1_ctrl` runs the policies and sends
motor commands. The simulator or real sensor drivers provide observations;
GLIM runs as a separate external program using the same configuration in sim/real.

Select localization explicitly: `glim` in sim or real, or `simulator` for
GLIM-free sim using robot-position truth. Marker recognition uses images,
measured joints and pelvis IMU with model kinematics, independent of GLIM/TF.
Evaluation object truth is not a runtime navigation input.

Start with the [individual startup workflow](doc/sim_real_workflow.md).
For sending motions, sequencing transport, and Viser preview/live/replay, use
[TaskPromptRL's documentation](https://github.com/isri-aist/TaskPromptRL#documentation).
Enter Velocity **before** releasing simulated hang. The supplied G1 configuration
selects `velocity/v0`, `navigation/v2_path` and `mimic/v1_lift_walk_put`; keep each
ONNX artifact paired with its own `params/deploy.yaml`.

## 1. Train

### 1.1 Velocity Tracking

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

Available velocity tracking tasks:

  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

### 1.2 Motion Imitation

#### Prepare Motion Files

Generate a MotionPrompt NPZ with TaskPromptRL. The file must contain
`format_version`, `fps`, `q_ref`, and `foot_contact`. Convert it to the tracking
format used by this repository:

```bash
python scripts/convert_motion_prompt_npz.py \
  --robot g1 \
  --input-file /path/to/motion_prompt.npz \
  --output-path /tmp/policy_input.npz
```

The converter reads the input FPS from the NPZ and writes the body states and
joint states required for tracking. A converted `lift_walk_put` example is
included at `src/assets/motions/g1/lift_walk_put.npz`.

#### Train

Train with the included example:

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-files=src/assets/motions/g1/lift_walk_put.npz \
  --env.scene.num-envs=4096
```

Replace `--motion-files` with `/tmp/policy_input.npz` to train with the newly
converted TaskPromptRL motion. A directory may also be specified to train with
all NPZ files directly inside it.

During training, each reference now keeps its final frame for a randomized
2--5 seconds with zero reference velocity. Pushes remain active in that interval,
and the hold rewards keep the arms on the final pose while allowing the lower
body to step when the torso tilt or angular velocity indicates loss of balance.

Available tasks:

  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

### 1.3 Navigation

For G1 29-DoF trajectory navigation with one- and two-second receding-horizon
path-pose observations:

```bash
python scripts/train.py Unitree-G1-Navigation-Flat --env.scene.num-envs=4096
```

Task details and observation conventions are documented in
[`src/tasks/navigation/README.md`](src/tasks/navigation/README.md).

### 1.4 Common Training Options

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

The first argument (e.g., `Unitree-G1-Flat`) specifies the training task.
> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/<robot>_(velocity | tracking | navigation)/<date_time>/model_<iteration>.pt`

## 2. Play

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint-file=logs/rsl_rl/g1_velocity/<run>/model_<iteration>.pt
```

Motion imitation:

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1/lift_walk_put.npz \
  --checkpoint-file logs/rsl_rl/g1_tracking/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer native
```

Choose a checkpoint trained for the supplied reference format. `--motion-file` expects a converted policy-input NPZ, not the raw
MotionPrompt NPZ. Use `--device cpu` when CUDA is unavailable, or `--viewer viser`
for browser-based visualization on a machine without a desktop display.

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.

**Visualization**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

## Acknowledgements


This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): versatile humanoid motion tracking framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
