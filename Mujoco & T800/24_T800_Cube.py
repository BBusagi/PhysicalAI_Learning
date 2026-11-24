"""T800 RL walk controlled with the keyboard, plus a red 8 cm wooden cube in front of the robot.

The cube is added in an XML string that includes t800.xml, so t800.xml is not modified.
Click the viewer window first.

Up / Down      forward speed vx   +/- 0.1 m/s
Left / Right   turn (yaw rate)    +/- 0.2 rad/s
Space          pause / resume

In the passive viewer these keys have no built-in function (W/A/S/D would toggle display options).
"""

import time
from collections import deque
from pathlib import Path

import MNN
import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
# Scene XML built in memory: include the original robot file, then add the cube.
# size = half-size (0.04 -> 8 cm edge), density 600 kg/m^3 = wood, 0.6 m in front resting on the floor
SCENE_XML = f"""
<mujoco>
  <include file="{HERE / 't800.xml'}"/>
  <worldbody>
    <body name="cube" pos="0.6 0 0.04">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" rgba="1 0 0 1" density="600"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(SCENE_XML)
data = mujoco.MjData(model)
policy = MNN.nn.load_module_from_file(str(HERE / "t800_260618_165257_30000.mnn"), [], [])

command = np.zeros(3)  # vx [m/s], vy [m/s], yaw rate [rad/s]  <- changed by the keyboard
CMD_MIN, CMD_MAX = [-0.6, -0.4, -1.0], [1.0, 0.4, 1.0]  # range the policy was trained on
KEYS = {265: ("Up", 0, +0.1), 264: ("Down", 0, -0.1), 263: ("Left", 2, +0.2), 262: ("Right", 2, -0.2)}  # GLFW key codes
SPACE = 32
paused = False

# Parameters from the SDK config rl_walking_example/default.yaml (25 joints, model order)
Q_DEFAULT = np.array([-0.06, 0, 0, 0.12, -0.06, 0] * 2 + [0] + [0, 0.15, 0, -0.25, 0] + [0, -0.15, 0, -0.25, 0] + [0, 0])
KP = np.array([180, 100, 100, 180, 40, 40] * 2 + [100] + [60, 50, 50, 60, 50] * 2 + [100, 100], dtype=float)
KD = np.array([5, 3, 3, 5, 0.3, 0.3] * 2 + [5] + [0.3] * 10 + [1, 1])
ACTION_SCALE = np.array([0.5, 0.2, 0.2, 0.5, 0.5, 0.2] * 2 + [0.2, 0.2, 0.05, 0.2, 0.05] * 2)
ACTIVE = [i for i in range(25) if i not in (12, 23, 24)]  # policy skips torso yaw and head
QPOS, QVEL = slice(7, 32), slice(6, 31)  # robot joints only (the cube's free joint comes after them)
HISTORY = 15  # frames of observation fed to the policy
DECIMATION = 5  # policy every 5 sim steps = 10 ms

gyro = data.sensor("imu_angular_velocity").data
quat = data.sensor("imu_quaternion").data
history = deque(maxlen=HISTORY)
last_action = np.zeros(len(ACTIVE))
q_target = Q_DEFAULT.copy()


def policy_step():
    global last_action, q_target
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    gravity = -R.reshape(3, 3).T @ [0, 0, 1]
    q, qd = data.qpos[QPOS], data.qvel[QVEL]
    obs = np.concatenate([(q - Q_DEFAULT)[ACTIVE], qd[ACTIVE] * 0.05, last_action, gyro, gravity])
    if not history:
        history.extend([obs] * HISTORY)
    history.append(obs)
    x = np.concatenate([*history, command * [2.0, 2.0, 1.0]]).astype(np.float32)  # 1083
    y = policy.forward(MNN.expr.const(x[None], [1, x.size], MNN.expr.NCHW))
    last_action = np.array(y.read()).ravel()
    q_target = Q_DEFAULT.copy()
    q_target[ACTIVE] += last_action * ACTION_SCALE


def control(model, data):
    if round(data.time / model.opt.timestep) % DECIMATION == 0:
        policy_step()
    data.ctrl[:] = KP * (q_target - data.qpos[QPOS]) - KD * data.qvel[QVEL]


def on_key(key):
    global paused
    if key in KEYS:
        name, i, delta = KEYS[key]
        command[i] = np.clip(command[i] + delta, CMD_MIN[i], CMD_MAX[i])
        print(f"{name:5s} -> command: vx={command[0]:+.1f} m/s  vy={command[1]:+.1f} m/s  yaw={command[2]:+.1f} rad/s")
    elif key == SPACE:
        paused = not paused
        print("paused" if paused else "resumed")


data.qpos[QPOS] = Q_DEFAULT
mujoco.set_mjcb_control(control)

with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    start = time.time()  # wall-clock time at which data.time was 0
    while viewer.is_running():
        if paused:
            start = time.time() - data.time  # shift the clock so sim time stays frozen
        else:
            while data.time < time.time() - start:  # step physics until it catches up with real time
                mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1 / 60)
