# Unitree RL Mjlab


## ✳️ Overview
Unitree RL Mjlab is a reinforcement learning project built upon the
[mjlab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its 
physics simulation backend, currently supporting Unitree Go2, A2, As2, G1, R1, H1_2 and H2.

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.


## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical Unitree robots for real-world execution.


## 🛠️ Usage Guide

### 1. Velocity Tracking Training

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

For G1 29-DoF object-front navigation with a head-camera/ArUco relative-pose
observation:

```bash
python scripts/train.py Unitree-G1-Navigation-Flat --env.scene.num-envs=4096
```

Task details and observation conventions are documented in
[`src/tasks/navigation/README.md`](src/tasks/navigation/README.md).

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- The first argument (e.g., Mjlab-Velocity-Flat-Unitree-G1) specifies the training task.
Available velocity tracking tasks:
  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. Motion Imitation Training

Train a Unitree G1 to mimic reference motion sequences.

<div style="margin-left: 20px;">

#### 2.1 Prepare Motion Files

Prepare csv motion files in mjlab/motions/g1/ and convert them to npz format:

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz files will be stored at:**：`src/motions/g1/...`

#### 2.2 Training

After generating the NPZ file, launch imitation training:

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

Available tasks:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> For detailed motion imitation instructions, refer to the BeyondMimic documentation:
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. Simulation Validation

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

Motion imitation:
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.

**Visualization**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 4. Real Deployment

Before deployment, install the required communication tools:
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 4.1 Power On the Robot
Start the robot in suspended state and wait until it enters `zero-torque` mode.

#### 4.2 Enable Debug Mode
While in `zero-torque` mode, press `L2 + R2` on the controller. The robot will enter `debug mode` with joint damping enabled.

#### 4.3 Connect to the Robot
Connect your PC to the robot via Ethernet. Configure the network as:
- Address：`192.168.123.222`
- Netmask：`255.255.255.0`

Use `ifconfig` to determine the Ethernet device name for deployment.

#### 4.4 Compilation

Example: Unitree G1 velocity control.
Place `policy.onnx` and `policy.onnx.data` into: `deploy/robots/g1/config/policy/velocity/v0/exported`.
Then compile:

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

The G1 controller also includes the navigation policy at
`deploy/robots/g1/config/policy/navigation/v0`. Controller mappings are:

- `L2 + Up`: Passive to FixStand
- `R2 + A`: FixStand/Navigation to Velocity
- `R2 + B`: FixStand/Velocity to Navigation
- `L2 + B`: return to Passive

The Navigation target is the robot-base pose 1 m in front of the localization
pose captured when Navigation is entered. On the real robot, the controller
subscribes to `/glim_ros/odom` (`nav_msgs/msg/Odometry`) and uses its planar
position and yaw instead of integrating the issued velocity command. It waits
with a zero navigation command when the selected localization source is not
available, and stops when updates exceed `odometry_timeout` (0.5 s by default).
The source, topics, and timeout are configured in
`deploy/robots/g1/config/policy/navigation/v0/params/deploy.yaml`.

For sim2sim, the default `localization_source: auto` falls back to the MuJoCo
truth position published on `rt/sportmodestate`; yaw comes from the simulated
IMU in `rt/lowstate`. GLIM is preferred when both inputs are available. The
selected source is fixed until Navigation is exited, preventing a coordinate
frame change while walking. Set `localization_source` explicitly to `glim` or
`simulator` when automatic selection is not desired.

Build from a shell where the ROS 2 environment is sourced. The odometry pose's
child frame must have its +x axis aligned with the G1 base forward direction;
if GLIM tracks a sensor frame with a different mounting transform, publish a
base-aligned odometry pose before using it here. Test suspended and in
`unitree_mujoco` before running on the physical robot.

#### 4.5 Deployment

## 4.5.1 Simulation Deployment

Before deploying on the real robot, it is recommended to perform simulation deployment using [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
to prevent abnormal behaviors on the physical robot. This framework has already integrated it.

Build unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

Build the G1 controller for sim2sim without a ROS 2 runtime dependency:

```bash
cmake -S deploy/robots/g1 -B deploy/robots/g1/build_sim \
  -DG1_NAVIGATION_WITH_ROS2=OFF
cmake --build deploy/robots/g1/build_sim -j8
```

Launch the simulator (note that a gamepad must be connected):

```bash
./simulate/build/unitree_mujoco
```

You can select the corresponding robot in `simulate/config`

Launch the simulation control program:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

For the ROS-free build above, run
`deploy/robots/g1/build_sim/g1_ctrl --network=lo`. It uses simulator truth
localization and does not require sourcing a ROS 2 setup file.

### Live Navigation pose plot (sim2sim and real robot)

From the repository root, start the plotter before or after entering Navigation:

```bash
python3 scripts/plot_deploy_navigation.py
```

Each Navigation entry truncates and then streams
`deploy/robots/g1/log/navigation_pose.csv` at the policy rate. The CSV contains
the relative-position and velocity commands, raw policy actions, processed
joint commands, encoder joints, joint targets actually written to LowCmd,
projected gravity, and the localization position/quaternion. The
`slam_pose_*` columns contain GLIM odometry on the real robot and MuJoCo truth
in sim2sim; use the `source` column (`glim` or `simulator`) to distinguish them.
Quaternion columns are ordered `qx,qy,qz,qw`. `command_joint_*` is the
processed policy target in policy joint order, while `action_joint_*` is the
last exact `q` value written to the mapped LowCmd motor entry. Unavailable data
is written as `nan` without stopping the other telemetry.

The plotter opens a pose window and a joint window. The pose window shows
`command_rel_pos` approaching zero and the localization pose projected onto an
XY plane with heading markers. Every G1 joint is shown in a small subplot with
`command_joints`, `encoder_joints`, and `action_joints` overlaid. To display
only selected joints, pass their policy-order indices, for example:

```bash
python3 scripts/plot_deploy_navigation.py --joints 0 3 6 9 12
```

The title reports whether the data source is `simulator` or `glim`; missing or
stale localization is shown explicitly. This file-based interface is identical
for the ROS-free sim2sim build and ROS-enabled real-robot build, and keeps GUI
work outside the control process.

The path, enable flag, and flush interval can be changed with
`pose_log_path`, `pose_log_enabled`, and `pose_log_flush_interval` in the
Navigation `deploy.yaml`. If required, install plotting dependencies with
`python3 -m pip install matplotlib numpy`.

## 4.5.2 Real-Robot Deployment

Launch the control program on the real robot:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0
```

The real-robot GLIM build requires `G1_NAVIGATION_WITH_ROS2=ON` (the default).
Source the matching ROS 2 environment before configuring, building, and
running it, for example `source /opt/ros/jazzy/setup.bash`.

**Arguments**：
- `network`: The network interface used to connect to the robot. Use `lo` for simulation deployment, and `enp5s0` for the real robot(You can check it using the `ifconfig` command) 

</div>

**Deployment Results**：

| Go2                                                    | G1                                                    | H1_2           | G1_mimic                                           |
|--------------------------------------------------------|-------------------------------------------------------|----------------|----------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-mimic-real.gif" width="300"/> |


## 🎉  Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): versatile humanoid motion tracking framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
