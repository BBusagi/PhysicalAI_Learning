"""Visualize key world-frame information of T800 on top of its skeleton:
joint axes, link centers of mass, sensor sites, whole-body CoM and the support polygon.
"""

from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

ax = plt.figure(figsize=(8, 9)).add_subplot(projection="3d")

# 1. Skeleton: line from each body to its parent
for i in range(2, model.nbody):
    p, q = data.xpos[i], data.xpos[model.body_parentid[i]]
    ax.plot(*zip(q, p), color="lightgray", linewidth=3)

# 2. Joint axes: arrow at each joint anchor along its rotation axis
ax.quiver(*data.xanchor[1:].T, *data.xaxis[1:].T, length=0.08, color="royalblue", label="joint axis")

# 3. Center of mass of each link
ax.scatter(*data.xipos[1:].T, color="orange", s=15, label="link CoM")

# 4. Sensor sites (IMU, foot force/torque sensors)
ax.scatter(*data.site_xpos.T, color="green", marker="^", s=40, label="sensor site")

# 5. Whole-body CoM and its projection on the ground
com = data.subtree_com[1]
ax.scatter(*com, color="red", marker="*", s=250, label="whole-body CoM")
ax.plot([com[0]] * 2, [com[1]] * 2, [com[2], 0], "r--", linewidth=1)
ax.scatter(com[0], com[1], 0, color="red", s=30)

# 6. Support polygon: convex hull of both feet's collision boxes projected to the ground
pts = []
for name in ("LINK_FOOT_L", "LINK_FOOT_R"):
    b = model.body(name).id
    for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
        for s in product((-1, 1), repeat=3):
            corner = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ (np.array(s) * model.geom_size[g])
            pts.append(corner[:2])
pts = sorted(map(tuple, pts))


def cross(o, a, b):  # 2D cross product of (a - o) and (b - o)
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def half_hull(points):  # monotone chain convex hull
    h = []
    for p in points:
        while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
            h.pop()
        h.append(p)
    return h[:-1]


hull = np.array(half_hull(pts) + half_hull(pts[::-1]) + [pts[0]])
ax.plot(hull[:, 0], hull[:, 1], 0, color="purple", linewidth=2, label="support polygon")

ax.set_xlabel("x (forward)")
ax.set_ylabel("y (left)")
ax.set_zlabel("z (up)")
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(0, 1.6)
ax.view_init(elev=15, azim=35)
ax.legend(loc="upper left")
ax.set_title("T800 key world-frame information")

out = Path(__file__).resolve().parent.parent / "outputs" / "t800_world_info.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)

if matplotlib.get_backend().lower() != "agg":
    plt.show()
