"""RPO humanoid tasks."""

import gymnasium as gym

for task_id, cfg_name in (("RPO-Velocity-Flat-Play", "RpoFlatPlayEnvCfg"), ("RPO-Maze-Hard", "RpoMazeHardEnvCfg")):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}"},
    )
