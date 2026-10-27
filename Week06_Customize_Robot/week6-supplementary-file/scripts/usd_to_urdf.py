"""Export a single-articulation robot USD (as written by Isaac Sim) to URDF + OBJ meshes.

This is the kit-less stand-in for Isaac Sim's "USD to URDF Exporter" extension. It reads
the USD physics schema directly with ``pxr``:

* every rigid body  -> ``<link>`` with ``<inertial>``, ``<visual>`` and ``<collision>``
* every revolute joint -> ``<joint type="revolute">`` with origin, axis and limits

Inertia: Isaac Sim USDs often leave ``physics:diagonalInertia`` at 0 so the engine computes it
from the collision shapes. URDF needs explicit values, so we compute them from the convex hull
of each collision mesh, scaled to the link mass (uniform density).

Usage (from the project folder):
    uv run python scripts/usd_to_urdf.py <robot.usd> <out_dir>
    -> <out_dir>/<robot>.urdf and <out_dir>/meshes/*.obj
"""

import math
import os
import sys

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


def rpy(q: Gf.Quatd) -> tuple[float, float, float]:
    """Quaternion -> URDF roll, pitch, yaw (fixed-axis XYZ)."""
    m = np.array(Gf.Matrix3d(q.GetNormalized())).T  # Gf is row-vector: transpose to column form
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(m[2, 0]) < 0.999999:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll, yaw = math.atan2(-m[1, 2], m[1, 1]), 0.0
    return roll, pitch, yaw


def fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" if abs(float(x)) > 1e-9 else "0" for x in v)


def local_pose(prim: Usd.Prim, root: Usd.Prim) -> Gf.Matrix4d:
    """Pose of ``prim`` relative to ``root`` (both xformable)."""
    cache = UsdGeom.XformCache()
    return cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(root).GetInverse()


def origin_xml(mat: Gf.Matrix4d) -> str:
    t = mat.ExtractTranslation()
    q = mat.RemoveScaleShear().ExtractRotationQuat()
    return f'<origin xyz="{fmt(t)}" rpy="{fmt(rpy(Gf.Quatd(q)))}"/>'


def write_obj(mesh: UsdGeom.Mesh, path: str) -> trimesh.Trimesh:
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    faces, k = [], 0
    for c in counts:  # fan-triangulate polygons
        f = idx[k : k + c]
        faces += [[f[0], f[i], f[i + 1]] for i in range(1, c - 1)]
        k += c
    tm = trimesh.Trimesh(pts, np.array(faces), process=False)
    tm.export(path)
    return tm


def diffuse(mesh: Usd.Prim) -> tuple[float, float, float] | None:
    """Base color of the material bound to ``mesh`` (MDL ``diffuse_color_constant`` or preview ``diffuseColor``)."""
    material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    for prim in Usd.PrimRange(material.GetPrim()) if material else []:
        if prim.IsA(UsdShade.Shader):
            for name in ("diffuse_color_constant", "diffuseColor"):
                value = UsdShade.Shader(prim).GetInput(name).Get() if UsdShade.Shader(prim).GetInput(name) else None
                if value is not None:
                    return tuple(value)
    return None


def main(usd_path: str, out_dir: str) -> None:
    stage = Usd.Stage.Open(usd_path)
    robot = stage.GetDefaultPrim()
    name = robot.GetName()
    os.makedirs(f"{out_dir}/meshes", exist_ok=True)
    out = [f'<?xml version="1.0"?>', f'<robot name="{name}">']

    for body in [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]:
        mass = UsdPhysics.MassAPI(body).GetMassAttr().Get() or 1.0
        geo, hulls = [], []
        for kind in ("visual", "collision"):
            group = body.GetChild("visuals" if kind == "visual" else "collisions")
            for mesh in [p for p in Usd.PrimRange(group) if p.IsA(UsdGeom.Mesh)] if group else []:
                fname = f"{body.GetName()}_{kind}.obj"
                tm = write_obj(UsdGeom.Mesh(mesh), f"{out_dir}/meshes/{fname}")
                pose = local_pose(mesh, body)
                color = diffuse(mesh) if kind == "visual" else None
                material = (
                    f'\n      <material name="{body.GetName()}_color"><color rgba="{fmt(color)} 1"/></material>'
                    if color
                    else ""
                )
                geo.append(
                    f'    <{kind}>\n      {origin_xml(pose)}\n      <geometry><mesh filename="meshes/{fname}"'
                    f' scale="1 1 1"/></geometry>{material}\n    </{kind}>'
                )
                if kind == "collision":
                    hull = tm.convex_hull
                    hull.apply_transform(np.array(pose).T)
                    hulls.append(hull)
        # inertia of the collision convex hulls, scaled to the link mass
        if hulls:
            hull = trimesh.util.concatenate(hulls).convex_hull
            hull.density = mass / hull.volume
            com, inertia = hull.center_mass, hull.moment_inertia
        else:
            com, inertia = np.zeros(3), np.eye(3) * 1e-4
        out += [
            f'  <link name="{body.GetName()}">',
            "    <inertial>",
            f'      <origin xyz="{fmt(com)}" rpy="0 0 0"/>',
            f'      <mass value="{mass:.6g}"/>',
            f'      <inertia ixx="{inertia[0,0]:.6g}" ixy="{inertia[0,1]:.6g}" ixz="{inertia[0,2]:.6g}"'
            f' iyy="{inertia[1,1]:.6g}" iyz="{inertia[1,2]:.6g}" izz="{inertia[2,2]:.6g}"/>',
            "    </inertial>",
            *geo,
            "  </link>",
        ]

    for jp in [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]:
        j = UsdPhysics.RevoluteJoint(jp)
        parent = j.GetBody0Rel().GetTargets()[0].name
        child = j.GetBody1Rel().GetTargets()[0].name
        # USD: child pose = parent * J0 * inverse(J1), with Jn = (localPosN, localRotN).
        # URDF puts the joint frame on the child link, so origin = J0 * inverse(J1)
        # and the axis is expressed in the child frame (rotated by localRot1).
        j0 = Gf.Matrix4d().SetRotate(Gf.Quatd(j.GetLocalRot0Attr().Get())).SetTranslateOnly(Gf.Vec3d(j.GetLocalPos0Attr().Get()))
        j1 = Gf.Matrix4d().SetRotate(Gf.Quatd(j.GetLocalRot1Attr().Get())).SetTranslateOnly(Gf.Vec3d(j.GetLocalPos1Attr().Get()))
        joint_origin = j1.GetInverse() * j0  # Gf matrices are row-vector: apply j1^-1 then j0
        unit = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[j.GetAxisAttr().Get()]
        axis = Gf.Rotation(Gf.Quatd(j.GetLocalRot1Attr().Get())).TransformDir(unit)
        lo, hi = math.radians(j.GetLowerLimitAttr().Get()), math.radians(j.GetUpperLimitAttr().Get())
        effort = jp.GetAttribute("drive:angular:physics:maxForce").Get() or 0.0
        vel = math.radians(jp.GetAttribute("physxJoint:maxJointVelocity").Get() or 0.0)
        out += [
            f'  <joint name="{jp.GetName()}" type="revolute">',
            f"    {origin_xml(joint_origin)}",
            f'    <parent link="{parent}"/>',
            f'    <child link="{child}"/>',
            f'    <axis xyz="{fmt(axis)}"/>',
            f'    <limit lower="{lo:.6g}" upper="{hi:.6g}" effort="{effort:.6g}" velocity="{vel:.6g}"/>',
            "  </joint>",
        ]
    out.append("</robot>")
    with open(f"{out_dir}/{name}.urdf", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {out_dir}/{name}.urdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
