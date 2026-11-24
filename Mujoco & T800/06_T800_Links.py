"""Print every link (body) of T800: its parent link and its mass."""

from pathlib import Path

import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))

for i in range(model.nbody):
    body = model.body(i)
    parent = model.body(body.parentid[0]).name
    print(f"{i:2d}  {body.name:22s} parent: {parent:22s} mass: {body.mass[0]:6.3f} kg")
