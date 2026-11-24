"""T800 pushes (kicks) a red 10 cm wooden cube forward - perception-based control.

Loop every 10 ms:
    perception : cube position in the robot frame (dx forward, dy left), from the simulator state
    planning   : keep facing world +x, side-step until the cube is in front of the right foot,
                 then walk forward and push it; step back if the cube got behind the feet
    command    : (vx, vy, yaw rate) -> RL walking policy -> joint targets -> PD
Space: pause / resume (click the viewer window first).
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
# size = half-size (0.05 -> 10 cm edge), density 600 kg/m^3 = wood, 0.6 m in front resting on the floor
SCENE_XML = f"""
<mujoco>
  <include file="{HERE / 't800.xml'}"/>
  <worldbody>
    <body name="cube" pos="0.6 0 0.05">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1" density="600"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(SCENE_XML)
data = mujoco.MjData(model)
policy = MNN.nn.load_module_from_file(str(HERE / "t800_260618_165257_30000.mnn"), [], [])

command = np.zeros(3)  # vx [m/s], vy [m/s], yaw rate [rad/s]  <- set by the planner
GOAL_YAW = 0.0  # push the cube along world +x
FOOT_Y = 0.12  # feet are 0.12 m left / right of the base: keep the cube in front of the RIGHT foot
VX, VX_BACK = 0.4, -0.3  # forward speed when lined up / backward speed when the cube got behind [m/s]
K_VY, VY_MIN, VY_MAX = 2.0, 0.2, 0.4  # side step: vy = K_VY * lateral error, magnitude in [VY_MIN, VY_MAX]
LAT_OK, LAT_GO = 0.03, 0.06  # lateral error [m]: no side step below LAT_OK, walk forward only below LAT_GO
K_YAW, YAW_MAX = 2.0, 0.8  # heading hold: yaw rate = K_YAW * heading error, limit [rad/s]
# The policy ignores speed commands below about 0.2 m/s (it just stands), hence VY_MIN.
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


def perceive():
    """Cube position in the robot (base) frame: dx forward, dy left [m], and the robot heading [rad]."""
    base_xy = data.qpos[0:2]
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    d = data.body("cube").xpos[:2] - base_xy  # world frame
    c, s = np.cos(yaw), np.sin(yaw)
    return c * d[0] + s * d[1], -s * d[0] + c * d[1], yaw  # rotate by -yaw


def plan(dx, dy, yaw):
    """Face +x, side-step until the cube is in front of the right foot, then walk forward and push it."""
    lat = dy + FOOT_Y  # lateral error: > 0 means the cube is left of the right foot
    heading_err = np.arctan2(np.sin(GOAL_YAW - yaw), np.cos(GOAL_YAW - yaw))  # wrapped to [-pi, pi]
    if dx < 0.05:  # cube is level with or behind the feet: step back
        command[0] = VX_BACK
    elif abs(lat) < LAT_GO:  # lined up: push
        command[0] = VX
    else:  # not lined up: stand and side-step first
        command[0] = 0.0
    command[1] = 0.0 if abs(lat) < LAT_OK else np.sign(lat) * np.clip(K_VY * abs(lat), VY_MIN, VY_MAX)
    command[2] = np.clip(K_YAW * heading_err, -YAW_MAX, YAW_MAX)


def control(model, data):
    step = round(data.time / model.opt.timestep)
    if step % DECIMATION == 0:
        dx, dy, yaw = perceive()
        plan(dx, dy, yaw)
        policy_step()
        if step % 250 == 0:  # print every 0.5 s
            print(f"t={data.time:5.1f}s  cube dx={dx:+.2f} dy={dy:+.2f} m  heading={np.degrees(yaw):+4.0f} deg  "
                  f"-> command vx={command[0]:.2f} vy={command[1]:+.2f} m/s  yaw={command[2]:+.2f} rad/s  "
                  f"cube xy=({data.body('cube').xpos[0]:.2f}, {data.body('cube').xpos[1]:+.2f}) m")
    data.ctrl[:] = KP * (q_target - data.qpos[QPOS]) - KD * data.qvel[QVEL]


def on_key(key):
    global paused
    if key == SPACE:
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
