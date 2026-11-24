"""Print the links, joints and motors of the T800 model."""

from pathlib import Path

import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))

print("== Links ==")
for i in range(model.nbody):
    print(i, model.body(i).name)

print("== Joints ==")
for i in range(model.njnt):
    print(i, model.joint(i).name)

print("== Motors ==")
for i in range(model.nu):
    print(i, model.actuator(i).name)
