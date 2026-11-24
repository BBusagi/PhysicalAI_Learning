"""Open the T800 model in the MuJoCo viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer

xml = Path(__file__).resolve().parent / "t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)  # blocks until the window is closed
