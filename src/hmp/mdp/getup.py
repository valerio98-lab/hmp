"""Fall-recovery helpers: resets from fallen mocap poses and get-up metrics.

``ResetFromClipGroundPose`` initializes a fraction of episodes from the
ground poses of the LaFAN fallAndGetUp clips, which are in-distribution for
the HMP (the fallen-reference initialization of the recovery task).
``got_up_frac`` and ``time_to_stand`` are logged metrics (not rewards).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = [
  "ResetFromClipGroundPose",
  "GetUpTracker",
  "got_up_frac",
  "time_to_stand",
]


def _tilt_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
  """Angle between the base's +z and world +z, from a (N,4) wxyz quaternion.

  ``projected_gravity_b.z`` = -(1 - 2(x^2 + y^2)); tilt = acos(-that).
  """
  x, y = quat_wxyz[:, 1], quat_wxyz[:, 2]
  gravity_z = -(1.0 - 2.0 * (x * x + y * y))
  return np.arccos(np.clip(-gravity_z, -1.0, 1.0))


class ResetFromClipGroundPose:
  """Reset event: put the robot in a random pose lying on the ground, taken
  from the fallAndGetUp mocap clips (root height and orientation plus joint
  angles), at rest.

  The pool of poses is built once, from every clip frame where the robot has
  settled on the ground: the base is tilted beyond ``limit_angle``, it is
  lower than ``max_height``, and it stays there for at least
  ``min_segment_s``. We need both conditions because tilt alone also picks up
  bent-over frames in the middle of a fall (about 6% of them, with the base
  still up to 0.70 m high). Those frames move fast in the clip, so spawning
  them at rest makes little physical sense and the robot just collapses. The
  median height of the poses we keep is 0.23 m.

  Root xy comes from the env origin (plus a bit of noise) rather than from the
  clip, whose position wanders around the capture volume; root z,
  orientation and joints come from the clip. Velocities are set to zero.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    params = cfg.params
    motion_dir = Path(params["motion_dir"])
    pattern = params.get("pattern", "*fallAndGetUp*.npz")
    limit_angle = float(params.get("limit_angle", np.radians(70.0)))
    min_segment_s = float(params.get("min_segment_s", 0.5))
    max_height = float(params.get("max_height", 0.45))

    files = sorted(motion_dir.glob(pattern))
    if not files:
      raise FileNotFoundError(f"No clips matching {pattern} in {motion_dir}")

    root_z, root_quat, joint_pos = [], [], []
    for path in files:
      data = np.load(path)
      fps = float(data["fps"][0])
      tilt = _tilt_from_quat(data["root_quat_w"])
      down = tilt > limit_angle
      if not down.any():
        continue
      idx = np.where(down)[0]
      segments = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
      min_len = int(min_segment_s * fps)
      for seg in segments:
        if len(seg) < min_len:
          continue
        keep = seg[data["root_pos"][seg, 2] < max_height]
        if keep.size == 0:
          continue
        root_z.append(data["root_pos"][keep, 2])
        root_quat.append(data["root_quat_w"][keep])
        joint_pos.append(data["joint_pos"][keep])

    if not root_z:
      raise RuntimeError("No ground poses found in the fallAndGetUp clips.")

    dev = env.device
    self.root_z = torch.tensor(np.concatenate(root_z), dtype=torch.float32, device=dev)
    self.root_quat = torch.tensor(
      np.concatenate(root_quat), dtype=torch.float32, device=dev
    )
    self.joint_pos = torch.tensor(
      np.concatenate(joint_pos), dtype=torch.float32, device=dev
    )
    self.num_poses = self.root_z.shape[0]
    print(
      f"[INFO] ResetFromClipGroundPose: {self.num_poses} ground poses "
      f"from {len(files)} clips (z {self.root_z.min():.2f}-{self.root_z.max():.2f})"
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    motion_dir: str,
    pattern: str = "*fallAndGetUp*.npz",
    limit_angle: float = 1.2217,
    min_segment_s: float = 0.5,
    max_height: float = 0.45,
    xy_range: tuple[float, float] = (-0.3, 0.3),
    prob: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del motion_dir, pattern, limit_angle, min_segment_s, max_height  # __init__
    if env_ids.numel() == 0:
      return
    # With ``prob`` < 1 the other envs keep the standing pose written by the
    # earlier reset events.
    if prob < 1.0:
      env_ids = env_ids[torch.rand(env_ids.numel(), device=env.device) < prob]
      if env_ids.numel() == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    n = env_ids.numel()
    dev = env.device

    pick = torch.randint(self.num_poses, (n,), device=dev)
    pos = env.scene.env_origins[env_ids].clone()
    pos[:, 0] += torch.empty(n, device=dev).uniform_(*xy_range)
    pos[:, 1] += torch.empty(n, device=dev).uniform_(*xy_range)
    pos[:, 2] = self.root_z[pick]

    asset.write_root_link_pose_to_sim(
      torch.cat([pos, self.root_quat[pick]], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=dev), env_ids=env_ids
    )
    joint_pos = self.joint_pos[pick]
    asset.write_joint_state_to_sim(
      joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
    )


class GetUpTracker:
  """Per-env get-up bookkeeping, shared by the metrics terms.

  ``stood`` latches once the base is upright and above ``stand_height`` for
  ``hold_s`` continuous seconds; ``t_to_stand`` records when. Success is a
  metric, not a termination: ending the episode on success would forfeit the
  remaining upright reward and make getting up pay less than lying still.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    n, dev = env.num_envs, env.device
    self.stood = torch.zeros(n, dtype=torch.bool, device=dev)
    self.t_to_stand = torch.zeros(n, device=dev)
    self._hold = torch.zeros(n, device=dev)
    self._elapsed = torch.zeros(n, device=dev)
    self._last_step = -1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.stood[env_ids] = False
    self.t_to_stand[env_ids] = 0.0
    self._hold[env_ids] = 0.0
    self._elapsed[env_ids] = 0.0

  def update(
    self,
    env: ManagerBasedRlEnv,
    stand_height: float,
    stand_tilt: float,
    hold_s: float,
    asset_cfg: SceneEntityCfg,
  ) -> None:
    # Both metric terms call this in the same step; only advance once.
    if env.common_step_counter == self._last_step:
      return
    self._last_step = env.common_step_counter

    asset: Entity = env.scene[asset_cfg.name]
    gravity_z = asset.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-gravity_z, -1.0, 1.0))
    up = (tilt < stand_tilt) & (asset.data.root_link_pos_w[:, 2] > stand_height)

    self._elapsed += env.step_dt
    self._hold = torch.where(up, self._hold + env.step_dt, torch.zeros_like(self._hold))
    newly = (self._hold >= hold_s) & ~self.stood
    self.t_to_stand = torch.where(newly, self._elapsed, self.t_to_stand)
    self.stood = self.stood | newly


class _TrackerTerm:
  """Base for the metric terms sharing one GetUpTracker per env.

  The tracker hangs off the env (not off the class) so two envs in one
  process can't silently share each other's buffers.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    tracker = getattr(env, "_getup_tracker", None)
    if tracker is None:
      tracker = GetUpTracker(cfg, env)
      env._getup_tracker = tracker  # type: ignore[attr-defined]
    self.tracker = tracker

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.tracker.reset(env_ids)


class got_up_frac(_TrackerTerm):
  """1 while the robot has stood up this episode, else 0.

  MetricsManager reports the per-step episode mean, i.e. the fraction of the
  episode spent after a successful get-up.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    stand_height: float = 0.6,
    stand_tilt: float = 0.5236,  # 30 deg
    hold_s: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    self.tracker.update(env, stand_height, stand_tilt, hold_s, asset_cfg)
    return self.tracker.stood.float()


class time_to_stand(_TrackerTerm):
  """Seconds from episode start to the first successful stand (0 if never)."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    stand_height: float = 0.6,
    stand_tilt: float = 0.5236,
    hold_s: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    self.tracker.update(env, stand_height, stand_tilt, hold_s, asset_cfg)
    return self.tracker.t_to_stand
