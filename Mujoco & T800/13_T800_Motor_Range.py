"""Loop over every T800 motor and print its control range (ctrlrange)."""

from pathlib import Path

import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))

for i in range(model.nu):
    motor = model.actuator(i)
    lo, hi = motor.ctrlrange
    print(f"{i:2d}  {motor.name:28s} ctrlrange: [{lo:7.1f}, {hi:7.1f}] N·m")
