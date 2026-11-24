"""Send ±10000 N·m to every T800 motor and print what the motors actually output.

MuJoCo clamps ctrl to ctrlrange internally, so the real output never exceeds the limit.
"""

from pathlib import Path

import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

CMD = 10000.0

print(f"{'motor':28s} {'ctrlrange':>18}   {'ctrl':>8} -> {'output':>7}   {'ctrl':>8} -> {'output':>7}")
for i in range(model.nu):
    lo, hi = model.actuator(i).ctrlrange

    data.ctrl[i] = CMD
    mujoco.mj_forward(model, data)  # computes actuator_force from ctrl
    out_pos = data.actuator_force[i]

    data.ctrl[i] = -CMD
    mujoco.mj_forward(model, data)
    out_neg = data.actuator_force[i]

    data.ctrl[i] = 0.0
    print(
        f"{model.actuator(i).name:28s} [{lo:7.1f}, {hi:7.1f}]   "
        f"{CMD:8.0f} -> {out_pos:7.1f}   {-CMD:8.0f} -> {out_neg:7.1f}"
    )

data.ctrl[0] = CMD
mujoco.mj_forward(model, data)
print(f"\nNote: data.ctrl[0] is still {data.ctrl[0]:.0f} -> MuJoCo clamps internally, it does not modify ctrl.")
