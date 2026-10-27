"""RPO humanoid (Roboparty) for Isaac Lab 3 + Newton.

PD gains, armature, effort limits and the standing pose are copied from Roboparty's own training config
(roboto_origin/modules/roboparty_train/robolab/robolab/assets/robots/roboparty.py), so their policy.onnx
sees the same motors it was trained with.
"""

from pathlib import Path

from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

PROJECT = Path(__file__).resolve().parents[2]
RPO_USD = str(PROJECT / "assets/usd/rpo/rpo.usda")  # made by scripts/urdf_to_usd.py

# joint order the Roboparty policy reads and writes (Isaac Sim / PhysX order: left, right, torso, ... interleaved)
POLICY_JOINTS = [
    "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
    "left_thigh_roll_joint", "right_thigh_roll_joint", "left_arm_pitch_joint", "right_arm_pitch_joint",
    "left_thigh_pitch_joint", "right_thigh_pitch_joint", "left_arm_roll_joint", "right_arm_roll_joint",
    "left_knee_joint", "right_knee_joint", "left_arm_yaw_joint", "right_arm_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_yaw_joint", "right_elbow_yaw_joint",
]  # fmt: skip

RPO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=RPO_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[NewtonArticulationCfg(self_collision_enabled=False)],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*_thigh_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.06,
            "right_arm_roll_joint": -0.06,
            ".*_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_thigh_yaw_joint", ".*_thigh_roll_joint", ".*_thigh_pitch_joint", ".*_knee_joint",
                              "torso_joint"],
            effort_limit=120.0,
            velocity_limit=25.0,
            stiffness={".*_thigh_.*": 100.0, ".*_knee_joint": 150.0, "torso_joint": 150.0},
            damping={".*_thigh_.*": 3.3, ".*_knee_joint": 5.0, "torso_joint": 5.0},
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=27.0,
            velocity_limit=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[".*_arm_pitch_joint", ".*_arm_roll_joint", ".*_arm_yaw_joint"],
            effort_limit=27.0,
            velocity_limit=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[".*_elbow_pitch_joint", ".*_elbow_yaw_joint"],
            effort_limit=27.0,
            velocity_limit=8.0,
            stiffness={".*_elbow_pitch_joint": 30.0, ".*_elbow_yaw_joint": 20.0},
            damping={".*_elbow_pitch_joint": 1.5, ".*_elbow_yaw_joint": 1.0},
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)
