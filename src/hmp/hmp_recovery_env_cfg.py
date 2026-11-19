"""Phase III fall-recovery velocity-tracking task (Sec. IV-A.3).

Same commands, observations, rewards and curriculum as velocity tracking, but
falling is a recoverable state instead of a terminal one:

1. ``failed_to_recover`` replaces ``fell_over``: the episode ends only if the
   robot stays down (tilt > 70 deg) for 10 s continuously.
2. ``reset_fallen``: 30% of episodes start from a fallen reference pose taken
   from the LaFAN ``fallAndGetUp`` clips (in-distribution for the HMP).
3. ``knockdown_robot``: from iteration 3500 on, strong randomized base-velocity
   perturbations (3-5 m/s) are applied to upright robots.
4. ``upright_recovery``: dense lying -> standing reward.
5. While down, the velocity command is zeroed (``RecoveryVelocityCommand``)
   and the tracking rewards only pay while genuinely standing.

The HMP must be distilled from a motion set that contains the fallAndGetUp
clips, otherwise the codebook has no get-up primitives.
"""

import math
from dataclasses import fields
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from hmp import mdp
from hmp.hmp_locomotion_env_cfg import (
  TWIST_COMMAND_NAME,
  make_locomotion_env_cfg,
)

# Source of the fallen reference poses used to initialize 30% of episodes.
FALL_MOTION_DIR = str(
  Path(__file__).resolve().parents[2] / "assets" / "motions" / "g1_lafan_recovery"
)

# Knockdown kicks: guaranteed-magnitude horizontal shove in a random direction.
KNOCKDOWN_SPEED_RANGE = (3.0, 5.0)
KNOCKDOWN_ANG_MAGNITUDE = 2.5


def make_recovery_env_cfg() -> ManagerBasedRlEnvCfg:
  """Velocity-tracking template + fall recovery (see module docstring)."""
  cfg = make_locomotion_env_cfg()

  # Zero the velocity command while the robot is down (tilt > 70 deg), so a
  # fallen robot is not asked to run; restored once it is up again.
  twist = cfg.commands[TWIST_COMMAND_NAME]
  assert isinstance(twist, UniformVelocityCommandCfg)
  cfg.commands[TWIST_COMMAND_NAME] = mdp.RecoveryVelocityCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)},
    down_limit_angle=math.radians(70.0),
  )

  for _t in ("track_linear_velocity", "track_angular_velocity"):
    cfg.rewards[_t].params["stand_height"] = 0.55
    cfg.rewards[_t].params["stand_tilt"] = math.radians(45.0)
  cfg.rewards["track_linear_velocity"].func = mdp.track_linear_velocity_upright
  cfg.rewards["track_angular_velocity"].func = mdp.track_angular_velocity_upright

  # Gentle balance pushes only for upright robots tracking a non-zero command.
  cfg.events["push_robot"].func = mdp.push_moving_upright_envs

  # Falling is allowed; only staying down for 10 s ends the episode (well
  # above the 1.3 s median get-up duration in the mocap clips).
  del cfg.terminations["fell_over"]
  cfg.terminations["failed_to_recover"] = TerminationTermCfg(
    func=mdp.FallenTooLong,
    params={
      "limit_angle": math.radians(70.0),
      "time_limit": 10.0,
    },
  )

  # 30% of resets start from a fallen pose of the fallAndGetUp clips.
  cfg.events["reset_fallen"] = EventTermCfg(
    func=mdp.ResetFromClipGroundPose,
    mode="reset",
    params={
      "motion_dir": FALL_MOTION_DIR,
      "pattern": "*fallAndGetUp*.npz",
      "min_segment_s": 0.5,
      "xy_range": (-0.3, 0.3),
      "prob": 0.3,
    },
  )

  # Strong perturbations from iteration 3500 on
  cfg.events["knockdown_robot"] = EventTermCfg(
    func=mdp.knockdown_upright_envs,
    mode="interval",
    interval_range_s=(8.0, 12.0),
    params={
      "speed_range": KNOCKDOWN_SPEED_RANGE,
      "ang_magnitude": KNOCKDOWN_ANG_MAGNITUDE,
      "limit_angle": math.radians(70.0),
      "start_step": 3500 * 24,
    },
  )

  cfg.rewards["upright_recovery"] = RewardTermCfg(
    func=mdp.upright_recovery,
    weight=0.5,
    params={"target_height": 0.72},
  )

  stand = {"stand_height": 0.6, "stand_tilt": 0.5236, "hold_s": 0.3}
  cfg.metrics = {
    "got_up_frac": MetricsTermCfg(func=mdp.got_up_frac, params=dict(stand)),
    "time_to_stand": MetricsTermCfg(func=mdp.time_to_stand, params=dict(stand)),
  }

  cfg.episode_length_s = 20.0
  return cfg
