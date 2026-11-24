"""Use a PD controller to hold T800 in its initial pose (all joint angles = 0).

    tau = kp * (q_target - q) - kd * qdot      (clipped to each motor's torque limit)
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

# Gains from the EngineAI SDK (assets/config/t800/pd_stand/default.yaml), in motor order:
# left leg (6) | right leg (6) | torso (1) | left arm (5) | right arm (5) | head (2)
kp = np.array([1080, 480, 880, 1000, 800, 100] * 2 + [200] + [120] * 10 + [100, 100], dtype=float)
kd = np.array([25, 25, 25, 25, 2, 2] * 2 + [1] + [1, 1, 1, 1, 0.2] * 2 + [1, 1], dtype=float)

q_target = data.qpos[7:].copy()  # initial joint angles (skip the 7 floating-base values)
lo, hi = model.actuator_ctrlrange.T  # torque limits

DURATION = 10.0
sync_every = round((1 / 60) / model.opt.timestep)
print_every = round(1.0 / model.opt.timestep)  # print once per second

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    wall_start = time.time()
    while viewer.is_running() and data.time < DURATION:
        q = data.qpos[7:]  # current joint angles
        qd = data.qvel[6:]  # current joint speeds
        data.ctrl[:] = np.clip(kp * (q_target - q) - kd * qd, lo, hi)

        mujoco.mj_step(model, data)
        step += 1

        if step % print_every == 0:
            print(f"t={data.time:5.1f}s  base height={data.qpos[2]:.3f} m  max |tau|={np.abs(data.ctrl).max():6.1f} N·m")

        if step % sync_every == 0:
            viewer.sync()
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)
