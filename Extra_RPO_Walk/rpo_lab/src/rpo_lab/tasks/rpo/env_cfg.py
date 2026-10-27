"""RPO tasks set up to run Roboparty's pretrained policy.onnx.

The policy reads 10 frames x 78 values, oldest frame first (frame-major):
    angular velocity (3), gravity in the body frame (3), command vx vy yaw (3),
    joint position minus standing pose (23), joint velocity (23), last action (23)
with joints in ``POLICY_JOINTS`` order, and outputs 23 joint offsets x 0.25 from the standing pose.

RPO-Velocity-Flat-Play   flat ground, driven with the keyboard (scripts/play_rpo.py)
RPO-Maze-Hard            the Week 7 Challenge 7.5 hazard maze (scripts/challenge_rpo.py)
"""

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

from g1_maze_lab.tasks.maze.env_cfg import G1MazeHardEnvCfg

from rpo_lab.robot import POLICY_JOINTS, RPO_CFG

ROBOT = SceneEntityCfg("robot", joint_names=POLICY_JOINTS, preserve_order=True)
# the URDF importer nests the links: Robot/Geometry/base_link/torso_link/...
TORSO_PRIM = "{ENV_REGEX_NS}/Robot/Geometry/base_link/torso_link"


def rpo_frame(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """One 78-value frame in the order the Roboparty policy was trained with."""
    return torch.cat(
        [
            mdp.base_ang_vel(env, asset_cfg),
            mdp.projected_gravity(env, asset_cfg),
            mdp.generated_commands(env, "base_velocity"),
            mdp.joint_pos_rel(env, asset_cfg),
            mdp.joint_vel_rel(env, asset_cfg),
            mdp.last_action(env),
        ],
        dim=-1,
    )


@configclass
class RpoObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # history 10, flattened: [frame t-9, ..., frame t] = 780 values
        frame = ObsTerm(func=rpo_frame, params={"asset_cfg": ROBOT}, history_length=10, flatten_history_dim=True)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def use_rpo(cfg):
    """Swap the G1 in a G1 velocity env config for the RPO and its pretrained policy's inputs / outputs."""
    cfg.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # the URDF importer nests the links, so the contact sensor has to match any depth
    cfg.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
    cfg.scene.height_scanner = None  # the RPO policy is blind: no height scan in its inputs
    cfg.observations = RpoObservationsCfg()
    # same as deploy: actions in POLICY_JOINTS order, target = standing pose + 0.25 x action
    cfg.actions.joint_pos.joint_names = POLICY_JOINTS
    cfg.actions.joint_pos.preserve_order = True
    cfg.actions.joint_pos.scale = 0.25
    cfg.actions.joint_pos.use_default_offset = True
    cfg.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.6)
    # play only: G1 body names in rewards / events do not exist on the RPO
    cfg.rewards = None
    cfg.curriculum = None
    cfg.events.add_base_mass = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.physics_material = None


@configclass
class RpoFlatPlayEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_rpo(self)
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 200
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 40
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # commands are set by the keyboard script
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}
        self.episode_length_s = 1.0e6


@configclass
class RpoMazeHardEnvCfg(G1MazeHardEnvCfg):
    """Challenge 7.5 hazard maze with the RPO. New maze: ``env.scene.terrain.terrain_generator.seed=<n>``."""

    def __post_init__(self):
        super().__post_init__()
        use_rpo(self)
        self.scene.maze_scanner.prim_path = TORSO_PRIM
        self.scene.wall_lidar.prim_path = TORSO_PRIM
        # a wall touching the arms or the body counts as a crash (G1: torso_link, pelvis, elbows, palms)
        self.terminations.body_contact.params["sensor_cfg"].body_names = ["base_link", "torso_link", ".*_elbow_.*"]
