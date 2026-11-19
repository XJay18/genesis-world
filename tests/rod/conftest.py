import math
import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def two_pad_gripper(request):
    """A gripper of two box pads facing each other along y, closing on a carriage that slides along z.

    The pads are tilted about x by the parameter of the fixture, in degrees (0 by default, set with `indirect=True`), so
    that the normals of their inner faces point down by that angle and their squeeze pushes a rope between them down.
    """
    tilt = math.radians(getattr(request, "param", 0.0))
    mjcf = ET.Element("mujoco", model="two_pad_gripper")
    worldbody = ET.SubElement(mjcf, "worldbody")
    carriage = ET.SubElement(worldbody, "body", name="carriage")
    ET.SubElement(carriage, "joint", name="lift", type="slide", axis="0 0 1")
    ET.SubElement(
        carriage, "geom", type="box", size="0.01 0.03 0.005", pos="0 0 0.04", mass="0.2", contype="0", conaffinity="0"
    )
    for side, sign in (("left", 1), ("right", -1)):
        pad = ET.SubElement(carriage, "body", name=f"{side}_pad", pos=f"0 {0.02 * sign} 0")
        ET.SubElement(pad, "joint", name=f"{side}_close", type="slide", axis=f"0 {-sign} 0")
        quat = f"{math.cos(0.5 * tilt)} {sign * math.sin(0.5 * tilt)} 0 0"
        ET.SubElement(pad, "geom", type="box", size="0.015 0.005 0.02", mass="0.1", quat=quat)
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture
def ball_on_slide():
    """A ball of 4 mm radius and 30 g on a vertical slide with viscous damping, to be pushed along it by a force."""
    mjcf = ET.Element("mujoco", model="ball_on_slide")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ball = ET.SubElement(worldbody, "body", name="ball")
    ET.SubElement(ball, "joint", name="push", type="slide", axis="0 0 1", damping="20")
    ET.SubElement(ball, "geom", type="sphere", size="0.004", mass="0.03")
    return ET.tostring(mjcf, encoding="unicode")
