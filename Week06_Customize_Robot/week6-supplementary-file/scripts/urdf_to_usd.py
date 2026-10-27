"""Convert a URDF to USD without Isaac Sim (kit-less), using Isaac Lab's UrdfConverter.

Needs the ``importers`` extra (``isaacsim-asset-isolated``), which this project installs.

Usage (from the project folder):
    uv run python scripts/urdf_to_usd.py <robot.urdf> <out_dir>
    -> <out_dir>/<robot>.usd
"""

import os
import shutil
import sys

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main(urdf: str, out_dir: str) -> None:
    # the importer writes <out_dir>/<name>_1 when <out_dir>/<name> exists, so start clean
    shutil.rmtree(os.path.join(out_dir, os.path.splitext(os.path.basename(urdf))[0]), ignore_errors=True)
    cfg = UrdfConverterCfg(
        asset_path=os.path.abspath(urdf),
        usd_dir=os.path.abspath(out_dir),
        usd_file_name=os.path.splitext(os.path.basename(urdf))[0] + ".usd",
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=True,
        force_usd_conversion=True,
        # PD gains come from the actuator config in Python, so leave the importer defaults
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"wrote {converter.usd_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
