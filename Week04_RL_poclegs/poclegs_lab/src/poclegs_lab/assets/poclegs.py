# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the poclegs biped (original-course robot, ported to Isaac Lab 3)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import POCLEGS_LAB_ASSETS_DIR

POCLEGS_USD = f"{POCLEGS_LAB_ASSETS_DIR}/poclegs/poclegs_newton.usd"
"""original-course USD, made Newton-friendly by ``scripts/make_newton_usd.py``."""
POCLEGS_URDF_USD = f"{POCLEGS_LAB_ASSETS_DIR}/usd_from_urdf/poclegs/poclegs.usda"
"""Same robot after USD -> URDF -> USD (``scripts/usd_to_urdf.py`` + ``scripts/urdf_to_usd.py``)."""

POCLEGS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=POCLEGS_USD,
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
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.449),
        joint_pos={
            "L[LR]_HR": 0.0,
            "L[LR]_HAA": -0.1745,
            "L[LR]_HFE": -0.1745,
            "L[LR]_KFE": 0.3491,
            "L[LR]_FFE": -0.1745,
        },
    ),
    actuators={
        "hr": DelayedPDActuatorCfg(
            joint_names_expr=[".*HR"],
            actuator_effort_limit=24.0,
            actuator_velocity_limit=23.0,
            stiffness=10.0,
            damping=1.5,
            armature=6.9e-5 * 81,
            friction=0.02,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "haa": DelayedPDActuatorCfg(
            joint_names_expr=[".*HAA"],
            actuator_effort_limit=30.0,
            actuator_velocity_limit=15.0,
            stiffness=15.0,
            damping=1.5,
            armature=9.4e-5 * 81,
            friction=0.02,
            min_delay=0,
            max_delay=4,
        ),
        "kfe": DelayedPDActuatorCfg(
            joint_names_expr=[".*HFE", ".*KFE"],
            actuator_effort_limit=30.0,
            actuator_velocity_limit=20.0,
            stiffness=15.0,
            damping=1.5,
            armature=1.5e-4 * 81,
            friction=0.02,
            min_delay=0,
            max_delay=4,
        ),
        "ffe": DelayedPDActuatorCfg(
            joint_names_expr=[".*FFE"],
            actuator_effort_limit=20.0,
            actuator_velocity_limit=23.0,
            stiffness=10.0,
            damping=1.5,
            armature=6.9e-5 * 81,
            friction=0.02,
            min_delay=0,
            max_delay=4,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Poclegs: 10-DoF biped, 17.5 kg, hip height ~0.45 m."""
