"""Print the pos of every body of T800: position relative to its parent body (from the XML)."""

from pathlib import Path

import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))


def fmt(v):
    return "[" + " ".join(f"{x:7.3f}" for x in v) + "]"  # fixed width per number


for i in range(model.nbody):
    print(f"{i:2d}  {model.body(i).name:22s} pos: {fmt(model.body(i).pos)}")
