"""MDP terms for the fall-recovery velocity-tracking task.

Falling is a recoverable state: the HMP codebook contains fall-and-get-up
primitives, so the task-level policy can learn to select them.

- ``FallenTooLong``: timed termination, only *staying* down fails.
- ``RecoveryVelocityCommand``: zero command while the robot is down.
- ``track_*_velocity_upright``: tracking reward only while standing.
- ``upright_recovery``: dense reward along the lying -> standing path.
- ``push_moving_upright_envs`` / ``knockdown_upright_envs``: perturbations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

from .events import push_moving_envs_by_setting_velocity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = [
  "FallenTooLong",
  "RecoveryVelocityCommand",
  "RecoveryVelocityCommandCfg",
  "knockdown_upright_envs",
  "push_moving_upright_envs",
  "track_linear_velocity_upright",
  "track_angular_velocity_upright",
  "upright_recovery",
]


def _standing_gate(
  env: ManagerBasedRlEnv,
  stand_height: float,
  stand_tilt: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """1 while the robot is really standing (tall and upright), 0 otherwise."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity_z = asset.data.projected_gravity_b[:, 2]
  tilt = torch.acos(torch.clamp(-gravity_z, -1.0, 1.0))
  tall = asset.data.root_link_pos_w[:, 2] > stand_height
  return (tall & (tilt < stand_tilt)).float()


def track_linear_velocity_upright(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  stand_height: float = 0.55,
  stand_tilt: float = 0.7854,  # 45 deg
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Linear-velocity tracking, paid only while genuinely standing.

  The tracking reward is a Gaussian on the velocity error, so a motionless
  robot lying under a zeroed command would collect it. Gating on standing
  (base height above ``stand_height`` and tilt below ``stand_tilt``) makes
  every lying/kneeling/propped pose earn zero tracking reward, leaving
  ``upright_recovery`` as the only signal while down. Height is the
  discriminator a tilt threshold alone cannot provide (a kneeling robot can
  have an upright torso); the defaults pass >99.9% of locomotion samples.
  """
  return vel_mdp.track_linear_velocity(
    env, std=std, command_name=command_name, asset_cfg=asset_cfg
  ) * _standing_gate(env, stand_height, stand_tilt, asset_cfg)


def track_angular_velocity_upright(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  stand_height: float = 0.55,
  stand_tilt: float = 0.7854,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Angular-velocity tracking, paid only while standing (see the linear
  version above)."""
  return vel_mdp.track_angular_velocity(
    env, std=std, command_name=command_name, asset_cfg=asset_cfg
  ) * _standing_gate(env, stand_height, stand_tilt, asset_cfg)


class RecoveryVelocityCommand(UniformVelocityCommand):
  """Velocity command forced to zero while the robot is down, restored on rise.

  A fallen robot conditioned on a live "run at 3 m/s" command selects running
  primitives instead of the get-up. While down (tilt > ``down_limit_angle``)
  the command is zero; once up, the sampled command is restored and tracking
  resumes (walk -> knocked down -> get up -> resume walking).
  """

  cfg: RecoveryVelocityCommandCfg

  def __init__(self, cfg: RecoveryVelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    # The base class zeroes ``vel_command_b`` in place for standing envs, so
    # we keep our own copy of the sampled command to restore it once the
    # robot is back on its feet.
    self._sampled_command_b = torch.zeros_like(self.vel_command_b)
    self.is_down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["down_frac"] = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._sampled_command_b[env_ids] = self.vel_command_b[env_ids]

  def _update_command(self) -> None:
    # Restore first, so a rise re-exposes the sampled command; then let the
    # base re-apply its own heading/standing logic on top.
    self.vel_command_b[:] = self._sampled_command_b
    super()._update_command()
    gravity_z = self.robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-gravity_z, -1.0, 1.0))
    self.is_down = tilt > self.cfg.down_limit_angle
    self.vel_command_b[self.is_down, :] = 0.0

  def _update_metrics(self) -> None:
    super()._update_metrics()
    # Fraction of episode time spent down, in [0, 1]. Accumulated like the
    # base class's error_vel_xy: the command manager logs the per-episode
    # value at reset and then zeroes the buffer. Unlike failed_to_recover it
    # also counts falls that are recovered.
    self.metrics["down_frac"] += (
      self.is_down.float() * self._env.step_dt / self._env.max_episode_length_s
    )


@dataclass(kw_only=True)
class RecoveryVelocityCommandCfg(UniformVelocityCommandCfg):
  """Base tilt beyond which the command is pinned to zero (70 deg = the same
  "down" measure used by FallenTooLong and the knockdown gate)."""

  down_limit_angle: float = 1.2217

  def build(self, env) -> RecoveryVelocityCommand:
    return RecoveryVelocityCommand(self, env)


class FallenTooLong:
  """End the episode only when the robot has been down for ``time_limit``
  seconds in a row (the per-env timer restarts as soon as it gets back up).

  "Down" uses the same measure as the velocity-tracking ``fell_over``
  termination (tilt of the gravity vector in the base frame beyond
  ``limit_angle``): a fall costs reward (tracking and upright terms pay ~0 on
  the ground) instead of ending the episode, which is what makes learning to
  get up possible.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._down_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._down_time[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    limit_angle: float,
    time_limit: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    gravity_z = asset.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-gravity_z, -1.0, 1.0))
    down = tilt > limit_angle
    self._down_time = torch.where(
      down,
      self._down_time + env.step_dt,
      torch.zeros_like(self._down_time),
    )
    return self._down_time > time_limit


def push_moving_upright_envs(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  command_name: str,
  command_threshold: float = 0.1,
  limit_angle: float = 1.2217,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Gentle balance pushes gated on both the command (moving envs only, see
  ``push_moving_envs_by_setting_velocity``) and posture (upright envs only,
  so a robot in the middle of a get-up is not pushed)."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity_z = asset.data.projected_gravity_b[env_ids, 2]
  tilt = torch.acos(torch.clamp(-gravity_z, -1.0, 1.0))
  upright_ids = env_ids[tilt < limit_angle]
  if upright_ids.numel() == 0:
    return
  push_moving_envs_by_setting_velocity(
    env,
    upright_ids,
    velocity_range=velocity_range,
    command_name=command_name,
    command_threshold=command_threshold,
    asset_cfg=asset_cfg,
  )


def knockdown_upright_envs(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  speed_range: tuple[float, float] = (3.0, 5.0),
  ang_magnitude: float = 2.5,
  z_magnitude: float = 0.3,
  limit_angle: float = 1.2217,  # 70 deg, same "down" measure as FallenTooLong
  start_step: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Strong knockdown shove, applied only to upright envs and only after
  ``start_step`` (second phase of the curriculum).

  The kick is a horizontal velocity impulse with magnitude sampled from
  ``speed_range`` in a random direction (plus a roll/pitch tip and a small
  upward pop), added to the base velocity. Sampling a direction and a
  guaranteed magnitude, rather than a symmetric ``+-v`` box whose probability
  mass concentrates near zero, makes every kick strong enough to cause a
  real fall.

  Phase gating: before ``start_step`` the policy learns gait and get-up from
  fallen resets with only the gentle balance pushes; afterwards the kicks
  teach it to get up from natural fall poses.

  Upright gating: a robot that is already down is not kicked, so get-up
  attempts are not disrupted.
  """
  if env.common_step_counter < start_step:
    return
  asset: Entity = env.scene[asset_cfg.name]
  gravity_z = asset.data.projected_gravity_b[env_ids, 2]
  tilt = torch.acos(torch.clamp(-gravity_z, -1.0, 1.0))
  upright_ids = env_ids[tilt < limit_angle]
  n = int(upright_ids.numel())
  if n == 0:
    return
  dev = env.device
  vel = asset.data.root_link_vel_w[upright_ids].clone()
  theta = torch.rand(n, device=dev) * (2.0 * math.pi)
  speed = torch.empty(n, device=dev).uniform_(*speed_range)
  vel[:, 0] += speed * torch.cos(theta)
  vel[:, 1] += speed * torch.sin(theta)
  vel[:, 2] += torch.empty(n, device=dev).uniform_(0.0, z_magnitude)
  vel[:, 3] += torch.empty(n, device=dev).uniform_(-ang_magnitude, ang_magnitude)
  vel[:, 4] += torch.empty(n, device=dev).uniform_(-ang_magnitude, ang_magnitude)
  asset.write_root_link_velocity_to_sim(vel, env_ids=upright_ids)


def upright_recovery(
  env: ManagerBasedRlEnv,
  target_height: float = 0.72,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense, monotone reward along the whole lying -> standing path.

  ``uprightness`` = (1 - g_z)/2 (1 upright, 0.5 lying, 0 upside down) times
  the base-height ratio clamped to [0, 1] (saturating slightly below the
  nominal 0.76 stand height so standing scores exactly 1, with no tip-toe
  incentive). It is close to linear rather than a Gaussian bowl, which would
  be flat at lying poses where the get-up needs a gradient. While upright it
  acts as a small constant alive bonus that is lost while on the ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  gravity_z = asset.data.projected_gravity_b[:, 2]
  uprightness = 0.5 * (1.0 - gravity_z)
  height = torch.clamp(asset.data.root_link_pos_w[:, 2] / target_height, 0.0, 1.0)
  return uprightness * height
