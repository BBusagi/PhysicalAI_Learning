"""Run the T800 simulation in the viewer for 10 s and print the base pos and rpy every 100 ms."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

PRINT_INTERVAL = 0.1  # s (simulated time)
DURATION = 10.0  # s, stop after 10 s of simulated time
steps_per_print = round(PRINT_INTERVAL / model.opt.timestep)  # 0.1 / 0.002 = 50 steps


def quat_to_rpy(q):
    """Quaternion [w x y z] -> roll, pitch, yaw [deg]."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


sync_every = round((1 / 60) / model.opt.timestep)  # refresh the window at ~60 fps (every 8 steps)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    wall_start = time.time()
    while viewer.is_running() and data.time < DURATION:
        mujoco.mj_step(model, data)
        step += 1

        if step % steps_per_print == 0:
            pos = data.qpos[0:3]  # base position x y z [m]
            rpy = quat_to_rpy(data.qpos[3:7])  # base orientation
            print(
                f"t={data.time:6.2f}s  "
                f"pos=[{pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}]  "
                f"rpy=[{rpy[0]:7.1f} {rpy[1]:7.1f} {rpy[2]:7.1f}] deg"
            )

        if step % sync_every == 0:
            viewer.sync()
            # Keep simulated time in step with wall-clock time: wait only if we are ahead.
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)

    print(f"simulated {data.time:.2f} s in {time.time() - wall_start:.2f} s of real time")
