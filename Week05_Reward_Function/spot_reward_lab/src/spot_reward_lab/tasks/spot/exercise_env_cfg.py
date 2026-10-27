# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Week 5 exercise configs (Exercise 5.2-5.4)."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg

from . import mdp as spot_mdp
from .tripod_env_cfg import SpotTripodEnvCfg

WORKING_FEET = "(fl|hl|hr)_foot"


@configclass
class SpotTripodFixEnvCfg(SpotTripodEnvCfg):
    """Exercise 5.3: the inherited Spot terms still score the broken foot (fr) - leave it out."""

    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        # Spot's air_time_reward hard-codes 4 feet (expand(-1, 4)) -> use our N-feet version
        r.air_time.func = spot_mdp.air_time_reward_any_feet
        r.air_time.params["sensor_cfg"].body_names = WORKING_FEET
        r.air_time_variance.params["sensor_cfg"].body_names = WORKING_FEET
        r.foot_slip.params["sensor_cfg"].body_names = WORKING_FEET
        r.foot_slip.params["asset_cfg"].body_names = WORKING_FEET
        r.foot_clearance.params["asset_cfg"].body_names = WORKING_FEET


@configclass
class SpotPaceEnvCfg(SpotFlatEnvCfg):
    """Exercise 5.2: pace = left pair and right pair move together."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.gait.params["synced_feet_pair_names"] = (("fl_foot", "hl_foot"), ("fr_foot", "hr_foot"))


def _limp_front_right(cfg):
    """Make the front-right leg really broken: no position control on fr_hx, fr_hy, fr_kn."""
    act = cfg.scene.robot.actuators
    act["spot_hip"].joint_names_expr = ["(fl|hl|hr)_h[xy]"]
    act["spot_knee"].joint_names_expr = ["(fl|hl|hr)_kn"]
    act["fr_limp_hip"] = act["spot_hip"].replace(joint_names_expr=["fr_h[xy]"], stiffness=0.0)
    act["fr_limp_knee"] = act["spot_knee"].replace(joint_names_expr=["fr_kn"], stiffness=0.0)


@configclass
class SpotLimpEnvCfg(SpotFlatEnvCfg):
    """Exercise 5.4: baseline task, but the fr motors have zero stiffness (only damping)."""

    def __post_init__(self):
        super().__post_init__()
        _limp_front_right(self)


@configclass
class SpotTripodLimpEnvCfg(SpotTripodEnvCfg):
    """Exercise 5.4: tripod task with the same limp fr leg."""

    def __post_init__(self):
        super().__post_init__()
        _limp_front_right(self)
