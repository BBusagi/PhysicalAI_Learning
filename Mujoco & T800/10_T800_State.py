"""Print the basic state of T800: sizes (nq, nv, nu) and the arrays qpos, qvel, ctrl."""

from pathlib import Path

import mujoco
import numpy as np

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

np.set_printoptions(precision=3, suppress=True, linewidth=100)

print("nq =", model.nq, "  (length of qpos)")
print("nv =", model.nv, "  (length of qvel)")
print("nu =", model.nu, "  (length of ctrl)")

print("\nqpos", data.qpos.shape, "\n", data.qpos)  # [x y z qw qx qy qz] + 25 joint angles
print("\nqvel", data.qvel.shape, "\n", data.qvel)  # [vx vy vz wx wy wz] + 25 joint speeds
print("\nctrl", data.ctrl.shape, "\n", data.ctrl)  # 25 motor torques
