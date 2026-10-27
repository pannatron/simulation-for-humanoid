"""Check that a URDF describes the same kinematics as the USD it came from.

For every joint it compares, at zero joint angles, the child link's position and orientation
(URDF forward kinematics vs USD body transforms) and the joint axis direction in world frame.
This is how we found the right-leg bug (``localRot1`` ignored -> right leg flipped 180 deg).

Usage (from the project folder):
    uv run python scripts/check_urdf.py <robot.usd> <robot.urdf>
"""

import sys
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


def main(usd_path: str, urdf_path: str) -> None:
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim().GetPath()
    cache = UsdGeom.XformCache()

    def usd_pose(name: str) -> np.ndarray:
        return np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{root}/{name}"))).T

    urdf = ET.parse(urdf_path).getroot()
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    base = next(link.get("name") for link in urdf.findall("link") if link.get("name") not in joints)
    poses = {base: usd_pose(base)}

    def fk(link: str) -> np.ndarray:
        if link not in poses:
            j = joints[link]
            o = j.find("origin")
            m = np.eye(4)
            m[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy").split()]).as_matrix()
            m[:3, 3] = [float(v) for v in o.get("xyz").split()]
            poses[link] = fk(j.find("parent").get("link")) @ m
        return poses[link]

    bad = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        j = UsdPhysics.RevoluteJoint(prim)
        parent = j.GetBody0Rel().GetTargets()[0].name
        child = j.GetBody1Rel().GetTargets()[0].name
        a, b = usd_pose(child), fk(child)
        pos_err = np.abs(a[:3, 3] - b[:3, 3]).max()
        rot_err = np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))
        q = j.GetLocalRot0Attr().Get()
        r0 = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        unit = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[j.GetAxisAttr().Get()]
        axis_usd = usd_pose(parent)[:3, :3] @ r0 @ unit
        urdf_joint = joints[child]
        axis_urdf = b[:3, :3] @ np.array([float(v) for v in urdf_joint.find("axis").get("xyz").split()])
        axis_err = np.degrees(np.arccos(np.clip(axis_usd @ axis_urdf, -1, 1)))
        ok = pos_err < 1e-4 and rot_err < 0.01 and axis_err < 0.01
        bad += not ok
        print(f"{prim.GetName():8s} pos_err={pos_err:.1e} m  rot_err={rot_err:6.2f} deg  axis_err={axis_err:6.2f} deg  {'OK' if ok else 'BAD'}")
    print(f"{bad} joint(s) differ" if bad else "all joints match")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
