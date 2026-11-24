# PhysicalAI_Learning

Personal notes and hands-on experiments while learning Physical AI / embodied intelligence:
robot models, simulation, control, and RL locomotion in MuJoCo with the EngineAI T800 humanoid.

## Repository layout

```
PhysicalAI_Learning/
├── Note.txt                  # Industrial robots vs. embodied intelligence
└── Mujoco & T800/            # MuJoCo × EngineAI T800 humanoid
    ├── 01_ ... 26_*.py       # Step-by-step learning scripts (see below)
    ├── Note.txt              # Study notes for this section
    ├── t800.xml              # Scene: T800 + ground (used by the scripts)
    ├── t800pro.xml           # Scene: T800 Pro + ground
    ├── environment/          # ground.xml, chair.xml
    ├── robot/                # EngineAI robot models   (NOT included, see below)
    └── t800_*.mnn            # EngineAI RL walk policy (NOT included, see below)
```

## Learning path

All scripts live in `Mujoco & T800/` and load `t800.xml`.

### 1. Basics

| Script | What it does |
|---|---|
| `01_helloworld.py` | MuJoCo hello world: a wooden block in free fall under gravity |
| `02_T800_Info.py` | Print the links, joints and motors of the T800 model |
| `03_T800_Dims.py` | Print the basic dimensions of the model (`nbody`, `njnt`, `nq`, `nv`, `nu`) |
| `04_T800_Viewer.py` | Open the T800 model in the MuJoCo viewer |

### 2. Model structure

| Script | What it does |
|---|---|
| `05_T800_Joints.py` | Every joint: which link it moves, its type and its range |
| `06_T800_Links.py` | Every link (body): its parent link and its mass |
| `07_T800_Body_Pos.py` | `pos` of every body: position relative to its parent |
| `08_T800_Body_Xpos.py` | `xpos` of every body: position in the world frame |
| `08b_T800_World_Info.py` | Key world-frame information drawn on top of the skeleton |
| `09_T800_Skeleton.py` | 3D skeleton plot from the world positions of the bodies |

### 3. State and control

| Script | What it does |
|---|---|
| `10_T800_State.py` | State sizes (`nq`, `nv`, `nu`) and the arrays `qpos`, `qvel`, `ctrl` |
| `11_T800_Sim_Print.py` | Run the simulation for 10 s in the viewer and print the base position |
| `12_T800_PD_Hold.py` | PD controller that holds the initial pose |
| `13_T800_Motor_Range.py` | Control range (`ctrlrange`) of every motor |
| `14_T800_Ctrl_Clamp.py` | Send ±10000 N·m to every motor and see what the motors actually apply |
| `15_T800_Manual_Joint.py` | Control the joints by hand with the viewer sliders |

### 4. RL walking

| Script | What it does |
|---|---|
| `17_T800_RL_Walk.py` | Walk with the official EngineAI RL policy (`.mnn`), ported from the SDK's C++ runner |
| `18_T800_RL_Walk_Simple.py` | Minimal version of 17 |
| `19_T800_RL_Walk_Log.py` | 18 plus a runtime log every 100 ms |
| `23_T800_Keyboard_Walk.py` | RL walk controlled with the keyboard |

How the policy is driven (from `17_T800_RL_Walk.py`):

```
Every 10 ms (policy step):
    obs (72)  = [q - q_default (22), qd * 0.05 (22), last action (22), gyro (3), gravity in body (3)]
    input     = last 15 obs (1080) + velocity command (3)  -> 1083
    action    = policy(input)                              -> 22
    q_target  = q_default + action * action_scale
Every 2 ms (sim step):
    tau = kp * (q_target - q) - kd * qd
```

### 5. Interaction with objects

| Script | What it does |
|---|---|
| `24_T800_Cube.py` | Keyboard walk with a red 8 cm wooden cube in the scene |
| `25_T800_Push_Cube.py` | Push (kick) a red 10 cm cube forward, using perception of the cube |
| `26_T800_Hold_Cube.py` | Keyboard walk while holding a red 10 cm cube in the right hand |

## Setup

Developed with Python 3.12 on WSL2 (Ubuntu 24.04).

```bash
pip install mujoco numpy matplotlib MNN
```

Versions used: mujoco 3.17.0, numpy 2.5.4, matplotlib 3.11.2, MNN 3.6.2.

## Third-party assets (not included)

The robot models and the RL policy belong to EngineAI (众擎机器人)
and are **not redistributed in this repository**. Get them from the EngineAI SDK and place them here:

```
Mujoco & T800/robot/t800/{meshes,texture,urdf,xml}/   # required by t800.xml
Mujoco & T800/t800_260618_165257_30000.mnn            # required by 17-26
```

Optional, for `t800pro.xml`: `Mujoco & T800/robot/t800pro/`. The repo was also explored with `robot/pm01_edu/`.

MuJoCo only needs the `.obj` meshes, `.png` textures and `xml` files. The `.dae` meshes are used by the URDF (ROS) only.

## How to run

```bash
cd "Mujoco & T800"
python 04_T800_Viewer.py         # look at the model
python 23_T800_Keyboard_Walk.py  # click the viewer first, then W/S speed, A/D turn, Enter stop
```

`08b` and `09` save their figures to `outputs/` at the repository root.

## Notes

Questions to ask about any robot learning setup (from `Mujoco & T800/Note.txt`):

1. Where are the models? Robot model (URDF for ROS, MJCF for MuJoCo, USD for Isaac) and policy model.
2. How is the state space defined? Proprioception, external, task, history, privileged information.
3. Is the controller PD or MPC?
4. What are the inputs and outputs of the policy?
5. Is the interaction visual servoing or force feedback?

Other topics in the notes: perception → planning → control, MNN inference (inspect models with Netron),
and four control approaches for the OpenArm robot arm (classic control, pre-trained + classic, VLA, task-specific).

## License

The code and notes in this repository are by the author.
EngineAI robot models and policies are subject to EngineAI's own terms and are not covered here.
