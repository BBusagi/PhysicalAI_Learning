"""Print the basic dimensions of the T800 model."""

from pathlib import Path

import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))

# MuJoCo has no "nmotor" field: count actuators of type <motor> (pure torque, no bias).
nmotor = sum(model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE for i in range(model.nu))

print("nbody  =", model.nbody)  # number of bodies (links), incl. world
print("njnt   =", model.njnt)  # number of joints
print("nmotor =", nmotor)  # number of torque motors
print("nq     =", model.nq)  # length of qpos (positions)
print("nv     =", model.nv)  # length of qvel (velocities / DoF)
print("nu     =", model.nu)  # number of actuators (length of ctrl)
