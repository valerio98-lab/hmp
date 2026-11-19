from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.envs import mdp as builtin_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .motion.base import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _rot6d_from_matrix(rotm: torch.Tensor) -> torch.Tensor:
  """Encode rotation matrix as [c1(3), c2(3)] (first two columns)."""
  first_two_cols = rotm[..., :, :2]
  return first_two_cols.transpose(-2, -1).reshape(rotm.shape[0], -1)


def _expand_model_field(
  current: torch.Tensor, default: torch.Tensor, num_envs: int
) -> torch.Tensor:
  if current.ndim == default.ndim:
    return current.unsqueeze(0).expand(num_envs, *current.shape)
  return current


def _resolve_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return _rot6d_from_matrix(matrix_from_quat(ori))


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return _rot6d_from_matrix(matrix_from_quat(ori_b))


def feet_contact_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def motion_friction_coeff(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  current_friction = _expand_model_field(
    env.sim.model.geom_friction,
    env.sim.get_default_field("geom_friction"),
    env.num_envs,
  )
  return current_friction[:, geom_ids, 0].mean(dim=-1, keepdim=True)


def motion_first_step_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  representation_name: str = "default",
) -> torch.Tensor:
  """Return the first per-step command from a stacked motion representation."""
  command = _resolve_command(env, command_name)
  command_obs = command.get_command_representation(representation_name)
  num_steps = max(len(command.future_sampling_step_offsets), 1)
  if num_steps <= 1 or command_obs.shape[-1] % num_steps != 0:
    return command_obs
  return command_obs.reshape(env.num_envs, num_steps, -1)[:, 0]


def _proprio_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Proprioception p_t (Eq. 1): [base_ang_vel, projected_gravity,
  joint_pos_rel, joint_vel_rel, last_action]."""
  return torch.cat(
    (
      builtin_mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel"),
      builtin_mdp.projected_gravity(env),
      builtin_mdp.joint_pos_rel(env),
      builtin_mdp.joint_vel_rel(env),
      builtin_mdp.last_action(env),
    ),
    dim=-1,
  )


def velocity_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return the body-frame velocity command ``(vx, vy, ω_z)``."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


class ProprioObservationHistory:
  """History of the proprioceptive observation p (Eq. 1) for H_alpha (Eq. 3).

  Returns the previous ``history_length`` proprio vectors flattened
  time-major as ``[p_{t-H}, ..., p_{t-1}]``. On reset the buffer is filled
  with the first post-reset observation.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.history_length = int(cfg.params["history_length"])
    if self.history_length <= 0:
      raise ValueError("ProprioObservationHistory requires history_length > 0.")
    self._history: torch.Tensor | None = None
    self._needs_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._history is None:
      return
    if env_ids is None or isinstance(env_ids, slice):
      self._needs_reset[:] = True
    else:
      self._needs_reset[env_ids] = True

  def __call__(self, env: ManagerBasedRlEnv, history_length: int) -> torch.Tensor:
    del history_length  # fixed at construction.
    current = _proprio_observation(env)

    if self._history is None:
      self._history = current[:, None, :].repeat(1, self.history_length, 1)
      self._needs_reset[:] = False
      return self._history.reshape(env.num_envs, -1)

    if torch.any(self._needs_reset):
      reset_ids = torch.nonzero(self._needs_reset, as_tuple=False).squeeze(-1)
      self._history[reset_ids] = current[reset_ids, None, :].repeat(
        1, self.history_length, 1
      )
      self._needs_reset[reset_ids] = False

    history_out = self._history.reshape(env.num_envs, -1).clone()
    self._history[:, :-1] = self._history[:, 1:].clone()
    self._history[:, -1] = current
    return history_out
