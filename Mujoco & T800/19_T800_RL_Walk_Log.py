"""T800 RL walk (based on 18) with a runtime log every 100 ms:
    [STATE]  base position and roll / pitch / yaw
    [POLICY] inference time and action statistics
    [PD]     torque, tracking error and torque saturation

Every 10 ms:  observation (72) x 15 frames + command (3) -> policy -> 22 joint targets
Every 2 ms:   PD control  tau = kp * (q_target - q) - kd * qd
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

COMMAND = np.array([0.3, 0.0, 0.0])  # vx [m/s], vy [m/s], yaw rate [rad/s]

# Parameters from the SDK config rl_walking_example/default.yaml (25 joints, model order)
Q_DEFAULT = np.array([-0.06, 0, 0, 0.12, -0.06, 0] * 2 + [0] + [0, 0.15, 0, -0.25, 0] + [0, -0.15, 0, -0.25, 0] + [0, 0])
KP = np.array([180, 100, 100, 180, 40, 40] * 2 + [100] + [60, 50, 50, 60, 50] * 2 + [100, 100], dtype=float)
KD = np.array([5, 3, 3, 5, 0.3, 0.3] * 2 + [5] + [0.3] * 10 + [1, 1])
ACTION_SCALE = np.array([0.5, 0.2, 0.2, 0.5, 0.5, 0.2] * 2 + [0.2, 0.2, 0.05, 0.2, 0.05] * 2)
ACTIVE = [i for i in range(25) if i not in (12, 23, 24)]  # policy skips torso yaw and head

ACTION_SIZE = 22  # one target offset per active joint
OBS_SIZE = 72  # joint pos 22 + joint vel 22 + last action 22 + gyro 3 + gravity 3
HISTORY = 15  # frames of observation fed to the policy
DECIMATION = 5  # policy every 5 sim steps = 10 ms
LOG_EVERY = 50  # log every 50 sim steps = 100 ms
TAU_LIMIT = model.actuator_ctrlrange[:, 1]  # motor torque limits

gyro = data.sensor("imu_angular_velocity").data  # live views into sensordata
quat = data.sensor("imu_quaternion").data

history = deque(maxlen=HISTORY)
last_action = np.zeros(ACTION_SIZE)
q_target = Q_DEFAULT.copy()
infer_ms = 0.0  # last inference time


def observation():
    q, qd = data.qpos[7:], data.qvel[6:]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    gravity = -R.reshape(3, 3).T @ [0, 0, 1]  # (0, 0, -1) when upright
    return np.concatenate([(q - Q_DEFAULT)[ACTIVE], qd[ACTIVE] * 0.05, last_action, gyro, gravity])


def policy_step():
    global last_action, q_target, infer_ms
    obs = observation()
    if not history:
        history.extend([obs] * HISTORY)  # first step: fill history with the current frame
    history.append(obs)
    x = np.concatenate([*history, COMMAND * [2.0, 2.0, 1.0]]).astype(np.float32)  # 1083
    t0 = time.perf_counter()
    y = policy.forward(MNN.expr.const(x[None], [1, x.size], MNN.expr.NCHW))
    last_action = np.array(y.read()).ravel()  # 22
    infer_ms = (time.perf_counter() - t0) * 1000
    q_target = Q_DEFAULT.copy()
    q_target[ACTIVE] += last_action * ACTION_SCALE


def control(model, data):
    """Called by MuJoCo at every sim step."""
    if round(data.time / model.opt.timestep) % DECIMATION == 0:
        policy_step()
    data.ctrl[:] = KP * (q_target - data.qpos[7:]) - KD * data.qvel[6:]
    if round(data.time / model.opt.timestep) % LOG_EVERY == 0:
        log()


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def log():
    pos, rpy = data.qpos[0:3], quat_to_rpy(data.qpos[3:7])
    tau = data.ctrl
    err = np.degrees(q_target - data.qpos[7:])  # PD tracking error
    worst = np.argmax(np.abs(tau) / TAU_LIMIT)  # motor closest to its limit
    print(
        f"t={data.time:6.2f}s\n"
        f"  [STATE]  pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:.3f}] m  "
        f"rpy=[{rpy[0]:+5.1f} {rpy[1]:+5.1f} {rpy[2]:+6.1f}] deg\n"
        f"  [POLICY] infer={infer_ms:.2f} ms  action min/max=[{last_action.min():+.2f} {last_action.max():+.2f}]  "
        f"|action| mean={np.abs(last_action).mean():.2f}\n"
        f"  [PD]     max|tau|={np.abs(tau).max():6.1f} N·m ({model.actuator(worst).name}, "
        f"{100 * abs(tau[worst]) / TAU_LIMIT[worst]:.0f}% of limit)  "
        f"max|err|={np.abs(err).max():5.2f} deg  saturated={int((np.abs(tau) > TAU_LIMIT).sum())}"
    )


data.qpos[7:] = Q_DEFAULT  # start in the default pose
mujoco.set_mjcb_control(control)  # MuJoCo calls control() inside every mj_step
mujoco.viewer.launch(model, data)  # the viewer runs the simulation in real time
