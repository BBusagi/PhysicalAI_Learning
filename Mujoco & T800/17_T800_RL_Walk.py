"""T800 walks with the official EngineAI RL policy (.mnn), re-implemented in Python.

Python port of the SDK's C++ runner:
    src/runner/rl_walking_example/src/rl_walking_example_runner.cc
    assets/config/t800/rl_walking_example/default.yaml

Every 10 ms (policy step):
    obs (72)  = [q - q_default (22), qd * 0.05 (22), last action (22), gyro (3), gravity in body (3)]
    input     = last 15 obs (1080) + velocity command * [2, 2, 1] (3)  -> 1083
    action    = policy(input)                                          -> 22
    q_target  = q_default + action * action_scale
Every 2 ms (sim step):
    tau = kp * (q_target - q) - kd * qd

Keyboard (in the viewer window):
    W / S   forward speed  +/- 0.1 m/s
    A / D   turn left / right  +/- 0.2 rad/s
    Enter   stop (all commands to 0)
"""

import time
from collections import deque
from pathlib import Path

import MNN
import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(HERE / "t800.xml"))
data = mujoco.MjData(model)
policy = MNN.nn.load_module_from_file(str(HERE / "t800_260618_165257_30000.mnn"), [], [])

# ---- parameters from rl_walking_example/default.yaml (25 joints in model order) ----
Q_DEFAULT = np.array([-0.06, 0, 0, 0.12, -0.06, 0] * 2 + [0] + [0, 0.15, 0, -0.25, 0] + [0, -0.15, 0, -0.25, 0] + [0, 0])
KP = np.array([180, 100, 100, 180, 40, 40] * 2 + [100] + [60, 50, 50, 60, 50] * 2 + [100, 100], dtype=float)
KD = np.array([5, 3, 3, 5, 0.3, 0.3] * 2 + [5] + [0.3] * 10 + [1, 1])
ACTION_SCALE = np.array([0.5, 0.2, 0.2, 0.5, 0.5, 0.2] * 2 + [0.2, 0.2, 0.05, 0.2, 0.05] * 2)
ACTIVE = [i for i in range(25) if i not in (12, 23, 24)]  # policy skips torso yaw and head
HISTORY = 15  # num_include_obs_steps

# ---- network sizes ----
ACTION_SIZE = len(ACTIVE)  # 22: one target offset per active joint
OBS_SIZE = 3 * ACTION_SIZE + 3 + 3  # 72 = joint pos + joint vel + last action + gyro + gravity
CMD_SIZE = 3  # vx, vy, yaw rate
INPUT_SIZE = OBS_SIZE * HISTORY + CMD_SIZE  # 1083 = network input
CONTROL_DT = 0.01
CMD_MAX = np.array([1.0, 0.4, 1.0])  # vx, vy, yaw rate (command_scale_pos)
CMD_MIN = np.array([-0.6, -0.4, -1.0])  # command_scale_neg
LPF_ALPHA = CONTROL_DT / (CONTROL_DT + 1 / (2 * np.pi * 0.1))  # 0.1 Hz command low-pass

decimation = round(CONTROL_DT / model.opt.timestep)  # 5 sim steps per policy step
gyro_adr = model.sensor("imu_angular_velocity").adr[0]
quat_adr = model.sensor("imu_quaternion").adr[0]

command = np.zeros(3)  # target set by keyboard
command_lpf = np.zeros(3)  # smoothed command fed to the policy
last_action = np.zeros(ACTION_SIZE)
history = deque(maxlen=HISTORY)
q_target = Q_DEFAULT.copy()


def observation() -> np.ndarray:
    q, qd = data.qpos[7:], data.qvel[6:]
    gyro = data.sensordata[gyro_adr : gyro_adr + 3]  # angular velocity in IMU frame
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.sensordata[quat_adr : quat_adr + 4])
    gravity = -R.reshape(3, 3).T @ np.array([0, 0, 1])  # (0, 0, -1) when upright
    obs = np.concatenate([(q - Q_DEFAULT)[ACTIVE], qd[ACTIVE] * 0.05, last_action, gyro, gravity])
    assert obs.size == OBS_SIZE
    return np.clip(obs, -100, 100)


def policy_step() -> None:
    global last_action, q_target, command_lpf
    command_lpf += LPF_ALPHA * (command - command_lpf)
    obs = observation()
    if not history:  # first call: fill the whole history with the current observation
        history.extend([obs] * HISTORY)
    else:
        history.append(obs)
    x = np.concatenate([*history, command_lpf * [2.0, 2.0, 1.0]]).astype(np.float32)
    assert x.size == INPUT_SIZE
    y = policy.forward(MNN.expr.const(x[None], [1, x.size], MNN.expr.NCHW))
    last_action = np.clip(np.array(y.read()).ravel(), -100, 100)
    assert last_action.size == ACTION_SIZE
    q_target = Q_DEFAULT.copy()
    q_target[ACTIVE] += last_action * ACTION_SCALE


def key_callback(key: int) -> None:
    keys = {ord("W"): (0, 0.1), ord("S"): (0, -0.1), ord("A"): (2, 0.2), ord("D"): (2, -0.2)}
    if key in keys:
        idx, delta = keys[key]
        command[idx] += delta
    elif key == 257:  # Enter
        command[:] = 0
    np.clip(command, CMD_MIN, CMD_MAX, out=command)
    print(f"command: vx={command[0]:+.1f} m/s  yaw={command[2]:+.1f} rad/s")


# ---- initial state: default pose -----------------------------------------
data.qpos[7:] = Q_DEFAULT
mujoco.mj_forward(model, data)

# W/A/S/D are also viewer shortcuts (wireframe, auto-connect, shadow, static body).
# We undo those toggles every frame so the keys only drive the robot.
RND_FLAGS = [int(mujoco.mjtRndFlag.mjRND_WIREFRAME), int(mujoco.mjtRndFlag.mjRND_SHADOW)]
VIS_FLAGS = [int(mujoco.mjtVisFlag.mjVIS_AUTOCONNECT), int(mujoco.mjtVisFlag.mjVIS_STATIC)]

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print(__doc__.split("Keyboard")[1])
    rnd0 = viewer.user_scn.flags[RND_FLAGS].copy()  # display flags at startup
    vis0 = viewer.opt.flags[VIS_FLAGS].copy()
    step = 0
    wall_start = time.time()
    while viewer.is_running():
        if step % decimation == 0:
            policy_step()
        data.ctrl[:] = KP * (q_target - data.qpos[7:]) - KD * data.qvel[6:]
        mujoco.mj_step(model, data)
        step += 1

        if step % round(0.5 / model.opt.timestep) == 0:
            v = data.qvel[0:2]
            print(f"t={data.time:6.1f}s  height={data.qpos[2]:.3f} m  speed=[{v[0]:+.2f} {v[1]:+.2f}] m/s")

        if step % round((1 / 60) / model.opt.timestep) == 0:
            lag = (time.time() - wall_start) - data.time  # > 0: simulation is behind real time
            if lag > 0.02:
                continue  # skip drawing this frame, keep simulating to catch up
            with viewer.lock():
                viewer.user_scn.flags[RND_FLAGS] = rnd0  # undo W / S display toggles
                viewer.opt.flags[VIS_FLAGS] = vis0  # undo A / D display toggles
            viewer.sync()
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)
