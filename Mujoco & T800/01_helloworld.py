"""MuJoCo Hello World: a wooden block in free fall under gravity.

A 10 cm wooden cube starts with its center at a height of 1 m and falls
onto the floor. The block's height is printed every 200 ms of simulated time,
next to the analytical free-fall solution z = z0 - 1/2 * g * t^2.

Usage:
    source ~/venvs/openarm/bin/activate
    python tutorials/01_helloworld.py            # print only (no window)
    python tutorials/01_helloworld.py --viewer   # also open a viewer window
"""

import argparse
import time

import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# 1. Describe the world in MJCF (MuJoCo's XML format).
# ---------------------------------------------------------------------------
INITIAL_HEIGHT = 1.0  # m, height of the block's center at t = 0
HALF_SIZE = 0.05  # m, half edge length -> 10 cm cube
WOOD_DENSITY = 600  # kg/m^3, typical for pine

XML = f"""
<mujoco model="helloworld">
  <!-- 1 ms timestep, standard gravity pointing down the z axis -->
  <option timestep="0.001" gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <!-- Floor: an infinite plane at z = 0 -->
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- The wooden block. freejoint = 6 DoF, it moves freely in 3D -->
    <body name="block" pos="0 0 {INITIAL_HEIGHT}">
      <freejoint/>
      <geom name="block_geom" type="box" size="{HALF_SIZE} {HALF_SIZE} {HALF_SIZE}"
            density="{WOOD_DENSITY}" rgba="0.72 0.53 0.30 1"/>
    </body>
  </worldbody>
</mujoco>
"""

PRINT_INTERVAL = 0.2  # s, print every 200 ms of simulated time
SIM_DURATION = 2.0  # s, total simulated time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--viewer", action="store_true", help="open a viewer window")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 2. Build the model (static description) and data (dynamic state).
    # -----------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # compute derived quantities at t = 0

    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    g = -model.opt.gravity[2]
    steps_per_print = round(PRINT_INTERVAL / model.opt.timestep)  # 200 steps
    total_steps = round(SIM_DURATION / model.opt.timestep)

    print(f"block mass: {model.body_mass[block_id]:.3f} kg, g = {g} m/s^2")
    print(f"block bottom touches the floor when its center reaches z = {HALF_SIZE} m\n")
    print(f"{'time [s]':>8} | {'height [m]':>10} | {'theory [m]':>10} | {'vz [m/s]':>8}")
    print("-" * 46)

    def report() -> None:
        t = data.time
        # freejoint qpos = [x y z qw qx qy qz], qvel = [vx vy vz wx wy wz].
        # (data.xpos would lag one step: mj_step computes it before integrating.)
        z = data.qpos[2]  # height of the block's center
        vz = data.qvel[2]
        theory = max(INITIAL_HEIGHT - 0.5 * g * t**2, HALF_SIZE)  # stops at floor
        print(f"{t:8.1f} | {z:10.4f} | {theory:10.4f} | {vz:8.3f}")

    # -----------------------------------------------------------------------
    # 3. Step the simulation.
    # -----------------------------------------------------------------------
    def run(viewer=None) -> None:
        report()  # t = 0
        for step in range(1, total_steps + 1):
            step_start = time.time()
            mujoco.mj_step(model, data)
            if step % steps_per_print == 0:
                report()
            if viewer is not None:
                if not viewer.is_running():
                    return
                viewer.sync()
                # Sleep so the window plays back in (roughly) real time.
                time.sleep(max(0.0, model.opt.timestep - (time.time() - step_start)))

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(viewer)
            print("\nSimulation finished. Close the window to exit.")
            while viewer.is_running():
                time.sleep(0.05)
    else:
        run()


if __name__ == "__main__":
    main()
