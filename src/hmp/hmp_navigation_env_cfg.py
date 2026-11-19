"""Phase III point-goal navigation task (Sec. IV-A.3).

The task-level policy receives the target position relative to the robot and
is rewarded for reducing the robot-target distance while staying balanced. A
new goal is sampled whenever the current one is reached, so the robot keeps
moving from target to target.

Command: ``c = (x_b, y_b, dist)``, the goal in the heading frame.

Robot, observation layout (except the command term), randomization and
sensors are inherited from ``make_locomotion_env_cfg``.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as vel_mdp

from hmp import mdp
from hmp.hmp_locomotion_env_cfg import (
  HISTORY_LENGTH,
  PUSH_VELOCITY_RANGE,
  _privileged_terms,
  _proprio_actor_terms,
  _proprio_critic_terms,
  make_locomotion_env_cfg,
)

NAVIGATION_COMMAND_NAME = "navigate"


def _navigation_command_cfg() -> mdp.NavigationGoalCommandCfg:
  return mdp.NavigationGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(20.0, 30.0),
    debug_vis=True,
    distance_range=(6.0, 10.0),
    angle_range=(-math.pi, math.pi),
    reach_tol=0.35,
    pos_clip=20.0,
  )


def _command_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.navigation_command,
    params={"command_name": NAVIGATION_COMMAND_NAME},
  )


def _history_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.ProprioObservationHistory,
    params={"history_length": HISTORY_LENGTH},
  )


def _rewards() -> dict[str, RewardTermCfg]:
  return {
    "nav_velocity_tracking": RewardTermCfg(
      func=mdp.nav_velocity_tracking_exp,
      weight=2.0,
      params={
        "command_name": NAVIGATION_COMMAND_NAME,
        "std": 0.5,
        "cruise_speed": 2.5,
      },
    ),
    "nav_position_tracking": RewardTermCfg(
      func=mdp.nav_position_tracking_exp,
      weight=1.0,
      params={"command_name": NAVIGATION_COMMAND_NAME, "std": 0.5},
    ),
    # Face the goal -> natural forward gait instead of sidestepping to it.
    "nav_heading": RewardTermCfg(
      func=mdp.nav_heading_exp,
      weight=0.5,
      params={"command_name": NAVIGATION_COMMAND_NAME, "std": 1.0},
    ),
    "nav_success": RewardTermCfg(
      func=mdp.nav_success_bonus,
      weight=3.0,
      params={"command_name": NAVIGATION_COMMAND_NAME},
    ),
    "self_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.5,
      params={"sensor_name": "self_collision"},
    ),
    "back_lean_orientation_l2": RewardTermCfg(
      func=mdp.back_lean_flat_orientation_l2,
      weight=-5.0,
    ),
    "upper_body_posture": RewardTermCfg(
      func=vel_mdp.posture,
      weight=0.2,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_.*",
          ),
        ),
        "std": {
          r".*shoulder_pitch.*": 0.15,
          r".*shoulder_roll.*": 0.15,
          r".*shoulder_yaw.*": 0.1,
          r".*elbow.*": 0.15,
          r".*wrist.*": 0.3,
        },
      },
    ),
  }


def _curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "navigate_levels": CurriculumTermCfg(
      func=mdp.navigate_command_levels,
      params={
        "command_name": NAVIGATION_COMMAND_NAME,
        "stages": [
          {
            "step": 0,
            "distance_range": (1.0, 2.0),
            "angle_range": (-0.5, 0.5),
          },
          {
            "step": 1000 * 24,
            "distance_range": (1.0, 3.0),
            "angle_range": (-1.5, 1.5),
          },
          {
            "step": 2500 * 24,
            "distance_range": (1.5, 4.0),
            "angle_range": (-math.pi, math.pi),
          },
          {
            "step": 5000 * 24,
            "distance_range": (10.0, 12.0),
            "angle_range": (-math.pi, math.pi),
          },
        ],
      },
    ),
  }


def make_navigation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Point-goal navigation template: actor obs ``[c | p | history]``."""
  cfg = make_locomotion_env_cfg()

  # Swap the twist command for the navigation goal command.
  cfg.commands = {NAVIGATION_COMMAND_NAME: _navigation_command_cfg()}

  # Rebuild the obs groups: same proprio/history/privileged, new command term.
  actor_terms = {
    "command": _command_term(),
    **_proprio_actor_terms(),
    "history": _history_term(),
  }
  critic_terms = {
    "command": _command_term(),
    **_proprio_critic_terms(),
    "policy_history": _history_term(),
    **_privileged_terms(),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.events["push_robot"].func = mdp.push_moving_envs_by_robot_speed
  cfg.events["push_robot"].params = {
    "velocity_range": PUSH_VELOCITY_RANGE,
    "speed_threshold": 0.3,
  }

  cfg.rewards = _rewards()
  cfg.curriculum = _curriculum()
  cfg.episode_length_s = 20.0
  return cfg
