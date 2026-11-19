from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def push_moving_envs_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Push only the envs whose velocity command is non-negligible.

  Envs commanded to stand (|v_xy| + |w_z| <= threshold) are not pushed, so
  the policy does not learn reactive arm/torso motions while standing still.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed = torch.norm(command[env_ids, :2], dim=1) + torch.abs(command[env_ids, 2])
  moving_env_ids = env_ids[speed > command_threshold]
  if moving_env_ids.numel() == 0:
    return
  push_by_setting_velocity(
    env, moving_env_ids, velocity_range=velocity_range, asset_cfg=asset_cfg
  )


def push_moving_envs_by_robot_speed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  speed_threshold: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Push only the envs whose base is actually moving (|v_xy| + |w_z|)."""
  asset: Entity = env.scene[asset_cfg.name]
  lin_speed = torch.norm(asset.data.root_link_lin_vel_b[env_ids, :2], dim=1)
  ang_speed = torch.abs(asset.data.root_link_ang_vel_b[env_ids, 2])
  speed = lin_speed + ang_speed
  moving_env_ids = env_ids[speed > speed_threshold]
  if moving_env_ids.numel() == 0:
    return
  push_by_setting_velocity(
    env, moving_env_ids, velocity_range=velocity_range, asset_cfg=asset_cfg
  )
