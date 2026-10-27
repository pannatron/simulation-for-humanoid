# Exercise 4.3 + Week 6: paste at the END of env_cfg.py (after PoclegsFlatEnvCfg).

##
# Tuned recipe (Week 4 "make it walk"): the Unitree H1 reward set from Isaac Lab 3, adapted to poclegs
##


@configclass
class PoclegsTunedRewardsCfg(RewardsCfg):
    """H1-style rewards: reward single-stance steps instead of penalizing short ones."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
            "threshold": 0.4,
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
    undesired_contacts = None
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*FFE")}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_torques_l2 = None
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)


@configclass
class PoclegsTunedFlatEnvCfg(PoclegsFlatEnvCfg):
    """Poclegs that learns to walk in ~20 min on a laptop GPU (see the Week 4 exercises)."""

    rewards: PoclegsTunedRewardsCfg = PoclegsTunedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # poclegs' legs are spread along x and swing about x: its "forward" is the base +-y axis
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # the USD caps HAA/KFE at 1.57 rad/s (90 deg/s): use the actuator speed limits instead
        for name, limit in {"hr": 23.0, "haa": 15.0, "kfe": 20.0, "ffe": 23.0}.items():
            self.scene.robot.actuators[name].joint_velocity_limit = limit


@configclass
class PoclegsUrdfFlatEnvCfg(PoclegsTunedFlatEnvCfg):
    """Poclegs rebuilt from URDF (Week 6), trained with the tuned recipe. Swap the robot with Hydra overrides, e.g.

    ``env.scene.robot.spawn.usd_path=/path/robot.usda "env.scene.robot.init_state.pos=[0,0,0.539]"``
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = POCLEGS_URDF_USD
        # the URDF importer nests links (base/ll_hr/ll_haa/...), so match bodies at any depth
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
        # URDF has no collision groups: overlapping hip hulls would touch the base at spawn
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
