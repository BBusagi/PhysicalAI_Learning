"""Draw the T800 skeleton in 3D from the world positions (xpos) of its bodies."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mujoco

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

ax = plt.figure(figsize=(7, 9)).add_subplot(projection="3d")

for i in range(1, model.nbody):  # skip world (0)
    p = data.xpos[i]
    parent = model.body_parentid[i]
    if parent != 0:  # bone: line from parent to child
        q = data.xpos[parent]
        ax.plot([q[0], p[0]], [q[1], p[1]], [q[2], p[2]], color="steelblue", linewidth=3)
    ax.scatter(*p, color="crimson", s=20)  # joint / body origin
    ax.text(*p, f" {model.body(i).name.removeprefix('LINK_')}", fontsize=6)

ax.set_xlabel("x (forward)")
ax.set_ylabel("y (left)")
ax.set_zlabel("z (up)")
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(0, 1.6)
ax.view_init(elev=10, azim=30)
ax.set_title("T800 skeleton (world positions)")

out = Path(__file__).resolve().parent.parent / "outputs" / "t800_skeleton.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)

if matplotlib.get_backend().lower() != "agg":  # interactive window if a GUI backend exists
    plt.show()
