"""Print the xpos of every body of T800: position in the world frame (forward kinematics)."""

from pathlib import Path

import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)  # compute world positions (data.xpos) from qpos


def fmt(v):
    return "[" + " ".join(f"{x:7.3f}" for x in v) + "]"  # fixed width per number


for i in range(model.nbody):
    print(f"{i:2d}  {model.body(i).name:22s} xpos: {fmt(data.body(i).xpos)}")
