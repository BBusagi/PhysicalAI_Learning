"""Print every joint of T800: which link it moves, its type and its range."""

from pathlib import Path

import mujoco
import numpy as np

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))

types = ["free", "ball", "slide", "hinge"]

for i in range(model.njnt):
    joint = model.joint(i)
    link = model.body(joint.bodyid[0]).name  # the link this joint moves
    jtype = types[joint.type[0]]
    if jtype == "hinge":
        lo, hi = np.degrees(joint.range)
        rng = f"{lo:7.1f} ~ {hi:6.1f} deg"
    else:
        rng = "-"
    print(f"{i:2d}  {joint.name or '(floating base)':22s} {jtype:6s} {link:22s} {rng}")
