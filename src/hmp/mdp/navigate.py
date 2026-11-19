"""Point-goal navigation command and rewards.

A ground goal is sampled around the robot; as soon as the base reaches it, a
new goal is sampled at least ``distance_range[0]`` metres away, so the robot
keeps moving from target to target (there is no stand/stop reward).

Command observed by the task-level policy (heading frame):
``[x_b, y_b, dist]``, the relative goal position plus the planar distance,
clipped. The frozen HMP never sees it; only the task-level policy does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = [
  "NavigationGoalCommand",
  "NavigationGoalCommandCfg",
  "navigation_command",
  "nav_velocity_tracking_exp",
  "nav_position_tracking_exp",
  "nav_heading_exp",
  "nav_success_bonus",
  "navigate_command_levels",
  "back_lean_flat_orientation_l2",
]


class NavigationGoalCommand(CommandTerm):
  """Single 2D ground goal; resamples the instant the base reaches it."""

  cfg: NavigationGoalCommandCfg

  def __init__(self, cfg: NavigationGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.distance = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.just_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Cumulative goals reached this episode (metric; auto-resets via metrics).
    self.reach_count = torch.zeros(self.num_envs, device=self.device)

    # Observed command: [x_b, y_b, dist] in the heading frame (clipped).
    self._command = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reach_count"] = self.reach_count

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _planar_distance(self) -> torch.Tensor:
    root_xy = self.robot.data.root_link_pos_w[:, :2]
    return torch.norm(root_xy - self.target_pos_w, dim=-1)

  def _refresh_command(self) -> None:
    """Goal in the heading frame: rotate (target - root) by -yaw."""
    root_xy = self.robot.data.root_link_pos_w[:, :2]
    yaw = self.robot.data.heading_w
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    dx = self.target_pos_w[:, 0] - root_xy[:, 0]
    dy = self.target_pos_w[:, 1] - root_xy[:, 1]
    x_b = cos_y * dx + sin_y * dy
    y_b = -sin_y * dx + cos_y * dy
    clip = self.cfg.pos_clip
    self._command[:, 0] = torch.clamp(x_b, -clip, clip)
    self._command[:, 1] = torch.clamp(y_b, -clip, clip)
    self._command[:, 2] = torch.clamp(self.distance, 0.0, clip)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    dist = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
    azim = torch.empty(n, device=self.device).uniform_(*self.cfg.angle_range)

    root_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    yaw = self.robot.data.heading_w[env_ids]
    direction = yaw + azim
    self.target_pos_w[env_ids, 0] = root_xy[:, 0] + dist * torch.cos(direction)
    self.target_pos_w[env_ids, 1] = root_xy[:, 1] + dist * torch.sin(direction)

    self.distance[env_ids] = torch.linalg.norm(
      self.robot.data.root_link_pos_w[env_ids, :2] - self.target_pos_w[env_ids],
      dim=-1,
    )
    self._refresh_command()

  def _update_command(self) -> None:
    self.distance = self._planar_distance()

    # Heading error: yaw vs bearing to the goal (for facing-the-goal reward).
    root_xy = self.robot.data.root_link_pos_w[:, :2]
    bearing = torch.atan2(
      self.target_pos_w[:, 1] - root_xy[:, 1],
      self.target_pos_w[:, 0] - root_xy[:, 0],
    )
    err = bearing - self.robot.data.heading_w
    self.heading_error = torch.atan2(torch.sin(err), torch.cos(err))

    self.reached = self.distance < self.cfg.reach_tol
    self.just_reached = self.reached.clone()
    reached_ids = self.reached.nonzero().flatten()
    if len(reached_ids) > 0:
      self.reach_count[reached_ids] += 1.0
      # New goal immediately (also resets the resampling timer).
      self._resample(reached_ids)
      self.distance = self._planar_distance()
    self._refresh_command()

  def _update_metrics(self) -> None:
    self.metrics["distance"] = self.distance
    self.metrics["reach_count"] = self.reach_count

  # -- Debug visualization -------------------------------------------------

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    target = self.target_pos_w.cpu().numpy()
    root = self.robot.data.root_link_pos_w.cpu().numpy()
    reached = self.reached.cpu().numpy()
    for batch in env_indices:
      tgt3 = (float(target[batch, 0]), float(target[batch, 1]), 0.05)
      color = (0.2, 0.9, 0.2, 0.9) if reached[batch] else (0.1, 0.5, 0.9, 0.9)
      visualizer.add_sphere(
        center=tgt3,
        radius=float(self.cfg.reach_tol),
        color=color,
        label=f"nav_target_{batch}",
      )
      visualizer.add_arrow(
        (float(root[batch, 0]), float(root[batch, 1]), 0.05),
        tgt3,
        color=(0.1, 0.4, 0.9, 0.6),
        width=0.01,
        label=f"nav_vec_{batch}",
      )


@dataclass(kw_only=True)
class NavigationGoalCommandCfg(CommandTermCfg):
  entity_name: str = "robot"

  distance_range: tuple[float, float] = (1.0, 2.0)
  """Planar spawn distance of a new goal from the robot (m). The minimum is
    strictly positive, so a freshly resampled goal is always some way off and
    the task cannot be satisfied by standing still."""

  angle_range: tuple[float, float] = (-0.5, 0.5)
  """Azimuth of the goal relative to the current heading (rad). Widened by the
    curriculum from a forward cone to full omnidirectional."""

  reach_tol: float = 0.35
  """Planar base-to-goal distance counted as 'reached' (m). On reach the goal
    immediately resamples."""

  pos_clip: float = 3.0
  """Clip (m) for the heading-frame goal position and distance in the command
    observation, so a far goal does not blow up the input scale."""

  def build(self, env: ManagerBasedRlEnv) -> NavigationGoalCommand:
    return NavigationGoalCommand(self, env)


def navigation_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return the heading-frame goal command ``[x_b, y_b, dist]``."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def _nav_command(env: ManagerBasedRlEnv, command_name: str) -> NavigationGoalCommand:
  command = env.command_manager.get_term(command_name)
  if isinstance(command, NavigationGoalCommand):
    return command
  raise TypeError(
    f"Command '{command_name}' is not a NavigationGoalCommand. Got: {type(command)}"
  )


def nav_velocity_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  cruise_speed: float = 1.5,
) -> torch.Tensor:
  """Reward tracking a velocity that points at the goal.

  Same Gaussian bowl as velocity tracking, but the target velocity is derived
  from the goal position instead of being an external command: a constant
  ``cruise_speed`` along the unit vector to the goal (world frame, no
  slowdown, so the robot keeps moving through the goal to the next one).
  Returns ``exp(-(||v_des - v_xy||^2 + v_z^2) / std^2)``. Peaking at a
  specific speed (rather than rewarding progress) favors a controlled gait
  over lunging; the z-velocity term discourages vertical bobbing. ``v_des`` is
  never observed by the policy."""
  command = _nav_command(env, command_name)
  root_xy = command.robot.data.root_link_pos_w[:, :2]
  to_goal = command.target_pos_w - root_xy
  dist = torch.norm(to_goal, dim=-1).clamp_min(1e-6)
  direction = to_goal / dist.unsqueeze(-1)
  v_des = direction * cruise_speed
  v_act = command.robot.data.root_link_lin_vel_w
  xy_error = torch.sum(torch.square(v_des - v_act[:, :2]), dim=-1)
  z_error = torch.square(v_act[:, 2])
  return torch.exp(-(xy_error + z_error) / std**2)


def nav_position_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Dense bowl around the goal: ``exp(-dist^2/std^2)``. Rewards actually
  stepping onto the goal in the last metre."""
  command = _nav_command(env, command_name)
  return torch.exp(-torch.square(command.distance) / std**2)


def nav_heading_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward facing the goal: ``exp(-heading_err^2/std^2)``. Encourages walking
  forward toward the target (natural gait) rather than sidestepping/moonwalking
  to it."""
  command = _nav_command(env, command_name)
  return torch.exp(-torch.square(command.heading_error) / std**2)


def nav_success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Sparse +1 on the step the base reaches the goal (pre-resample)."""
  command = _nav_command(env, command_name)
  return command.just_reached.float()


def back_lean_flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize leaning backward: squared x-component of the projected gravity
  when it is negative (base tilted back)."""
  asset: Entity = env.scene[asset_cfg.name]
  back_lean = asset.data.projected_gravity_b[:, 0]
  back_lean_mask = back_lean < 0.0
  rew = torch.where(
    back_lean_mask, torch.square(back_lean), torch.zeros_like(back_lean)
  )
  return rew


class NavLevelStage(TypedDict, total=False):
  step: int
  distance_range: tuple[float, float]
  angle_range: tuple[float, float]


def navigate_command_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  command_name: str,
  stages: list[NavLevelStage],
) -> dict[str, float]:
  """Curriculum: widen the goal spawn distance and azimuth over training."""
  del env_ids
  command = cast(NavigationGoalCommand, env.command_manager.get_term(command_name))
  distance_range = command.cfg.distance_range
  angle_range = command.cfg.angle_range
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      distance_range = stage["distance_range"]
      angle_range = stage["angle_range"]
  command.cfg.distance_range = distance_range
  command.cfg.angle_range = angle_range
  return {
    "distance_max": float(distance_range[1]),
    "angle_max": float(angle_range[1]),
  }
