"""Lengthen (or shorten) the thigh and shank of the poclegs URDF.

A link's length is the joint ``<origin xyz>`` that places the next joint. For poclegs the leg
points along -Y of each leg link frame:

    LL_KFE origin (-0.0025, -0.14, 0)  -> thigh length 0.14 m (link ll_hfe)
    LL_FFE origin (-0.042,  -0.16, 0)  -> shank length 0.16 m (link ll_kfe)

Scaling a link by ``s`` therefore means:

1. multiply the *next* joint's origin by ``s`` (moves the knee / ankle down),
2. stretch that link's visual + collision mesh along Y by ``s``
   (``scale="1 s 1"`` and the mesh origin multiplied by ``s`` so it stretches about the joint).

The same edits can be made by hand in any text editor - this script just does them for both legs.

Usage (from the project folder):
    uv run python scripts/scale_leg.py <in.urdf> <out.urdf> --thigh 1.3 --shank 1.3
    uv run python scripts/scale_leg.py <in.urdf> <out.urdf> --shank 1.15 --side left
"""

import argparse
import xml.etree.ElementTree as ET

# link to stretch -> joints (left, right) whose origin sets that link's length
SEGMENTS = {
    "thigh": ({"ll_hfe", "lr_hfe"}, {"LL_KFE", "LR_KFE"}),
    "shank": ({"ll_kfe", "lr_kfe"}, {"LL_FFE", "LR_FFE"}),
}


def vec(text: str) -> list[float]:
    return [float(x) for x in text.split()]


def fmt(v: list[float]) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--thigh", type=float, default=1.0, help="thigh length factor")
    p.add_argument("--shank", type=float, default=1.0, help="shank length factor")
    p.add_argument("--scale_mass", action="store_true", help="scale link mass with its length")
    p.add_argument("--side", choices=["both", "left", "right"], default="both", help="which leg to change")
    a = p.parse_args()

    keep = {"both": ("l", "r"), "left": ("l",), "right": ("r",)}[a.side]  # ll_* / lr_*, LL_* / LR_*
    tree = ET.parse(a.src)
    robot = tree.getroot()
    added = 0.0
    for seg, (links, joints) in SEGMENTS.items():
        s = getattr(a, seg)
        if s == 1.0:
            continue
        for joint in robot.findall("joint"):
            if joint.get("name") in joints and joint.get("name")[1].lower() in keep:
                o = joint.find("origin")
                xyz = vec(o.get("xyz"))
                if joint.get("name")[1].lower() == keep[0]:
                    added += abs(xyz[1]) * (s - 1.0)
                o.set("xyz", fmt([xyz[0], xyz[1] * s, xyz[2]]))
        for link in robot.findall("link"):
            if link.get("name") not in links or link.get("name")[1] not in keep:
                continue
            for tag in ("visual", "collision"):
                for g in link.findall(tag):
                    o = g.find("origin")
                    xyz = vec(o.get("xyz"))
                    o.set("xyz", fmt([xyz[0], xyz[1] * s, xyz[2]]))
                    m = g.find("geometry/mesh")
                    sc = vec(m.get("scale", "1 1 1"))
                    m.set("scale", fmt([sc[0], sc[1] * s, sc[2]]))
            inertial = link.find("inertial")
            o = inertial.find("origin")
            xyz = vec(o.get("xyz"))
            o.set("xyz", fmt([xyz[0], xyz[1] * s, xyz[2]]))
            if a.scale_mass:
                mass = inertial.find("mass")
                mass.set("value", f"{float(mass.get('value')) * s:.6g}")
    tree.write(a.dst, xml_declaration=True, encoding="utf-8")
    print(f"wrote {a.dst}")
    print(f"{a.side} leg(s) ~{added:.3f} m longer -> raise init_state pos z from 0.449 to about {0.449 + added:.3f}")


if __name__ == "__main__":
    main()
