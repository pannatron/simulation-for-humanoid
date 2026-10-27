"""Make a Newton-friendly copy of a robot USD exported from Isaac Sim.

Isaac Sim's URDF importer writes collision geometry as *instanceable* Xforms that carry
``PhysicsCollisionAPI`` while the actual ``Mesh`` lives inside the instance prototype.
PhysX accepts this, but Newton's USD importer only creates collision shapes for geometry
prims that carry the collision API themselves. This script:

1. flattens all sublayers / references into one file,
2. turns instancing off so every mesh becomes a regular prim,
3. moves ``PhysicsCollisionAPI`` + ``PhysicsMeshCollisionAPI`` (and the approximation,
   e.g. ``convexHull``) from the Xform onto each Mesh below it.

Usage (from the project folder):
    uv run python scripts/make_newton_usd.py <in.usd> <out.usd>
"""

import sys

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def main(src: str, dst: str) -> None:
    stage = Usd.Stage.Open(src)
    flat = Usd.Stage.Open(stage.Flatten())
    # 1) de-instance
    for prim in list(flat.Traverse()):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    # 2) move collision APIs onto the meshes
    moved = 0
    for prim in list(flat.Traverse()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Gprim):
            continue
        approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "convexHull"
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child)
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(approx)
                moved += 1
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
    flat.GetRootLayer().Export(dst)
    print(f"wrote {dst}: collision meshes={moved}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
