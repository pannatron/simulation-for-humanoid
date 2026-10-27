# Paste into env_cfg.py (Week 4 slides). Imports are already at the top of my_poclegs_env_cfg.py.

@configclass
class PoclegsRewardsCfg(RewardsCfg):
    """Reward terms for the MDP (original-course values)."""

    dof_acc_l2 = None
    # extra terms, off (weight 0) to match the original course - turn them on with Hydra overrides
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_range,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
            "command_name": "base_velocity",
            "threshold_min": 0.2,
            "threshold_max": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ffe"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*hfe", ".*haa"]), "threshold": 1.0},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"])},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*KFE"])},
    )
