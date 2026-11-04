import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def two_pad_gripper():
    """A gripper of two box pads facing each other along y, closing on a carriage that slides along z."""
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
        ET.SubElement(pad, "geom", type="box", size="0.015 0.005 0.02", mass="0.1")
    return ET.tostring(mjcf, encoding="unicode")
