"""Manually control T800 joints with the sliders in the viewer.

The model is modified in code (the XML files are not changed):
  - the free joint is removed, so the robot hangs fixed in the air and cannot fall
  - each torque motor becomes a position actuator: slider value = target joint angle [rad]

In the viewer: open the "Control" panel on the right and drag the sliders.
"""

from pathlib import Path

import mujoco
import mujoco.viewer

xml = Path(__file__).resolve().parent / "t800.xml"
spec = mujoco.MjSpec.from_file(str(xml))

# PD gains from the EngineAI SDK pd_stand config, in motor order
KP = [1080, 480, 880, 1000, 800, 100] * 2 + [200] + [120] * 10 + [100, 100]
KD = [25, 25, 25, 25, 2, 2] * 2 + [1] + [1, 1, 1, 1, 0.2] * 2 + [1, 1]

spec.delete(spec.body("LINK_BASE").joints[0])  # remove the free joint -> fixed base

for motor, kp, kd in zip(spec.actuators, KP, KD):
    torque_limit = motor.ctrlrange.copy()  # original ctrlrange = torque limit
    motor.set_to_position(kp=kp, kv=kd)  # ctrl now means target angle
    motor.ctrlrange = spec.joint(motor.target).range  # slider range = joint range [rad]
    motor.forcerange = torque_limit  # keep the real torque limit

model = spec.compile()
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
