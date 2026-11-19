"""Phase III velocity-tracking task (Sec. IV-A.3).

The task-level policy selects HMP codebook entries from the command
``c = (vx, vy, wz)`` (body frame), proprioception and the history embedding.
This template is also the base of the navigation and fall-recovery tasks.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from hmp import mdp

HISTORY_LENGTH = 10
TWIST_COMMAND_NAME = "twist"

PUSH_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def _velocity_command_kwargs() -> dict[str, object]:
  return {
    "entity_name": "robot",
    "resampling_time_range": (5.0, 10.0),
    "rel_standing_envs": 0.1,
    "rel_heading_envs": 0.0,
    "heading_command": False,
    "debug_vis": True,
    "ranges": UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
    ),
  }


def _proprio_actor_terms() -> dict[str, ObservationTermCfg]:
  """Same proprio block used by the HMP imitation actor, with corruption."""
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def _proprio_critic_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def _privileged_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "feet_contact_mask": ObservationTermCfg(
      func=mdp.feet_contact_mask,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "friction_coeff": ObservationTermCfg(
      func=mdp.motion_friction_coeff,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=())},
    ),
  }


def _command_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.velocity_command,
    params={"command_name": TWIST_COMMAND_NAME},
  )


def _history_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.ProprioObservationHistory,
    params={"history_length": HISTORY_LENGTH},
  )


def _actions() -> dict[str, ActionTermCfg]:
  """Same action space as the HMP: a = q_nom + scale * action.

  The per-joint scale is set per robot (G1_ACTION_SCALE) in
  config/g1/env_cfgs.py.
  """
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }


def _events() -> dict[str, EventTermCfg]:
  return {
    "reset_base": EventTermCfg(
      func=vel_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=vel_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_moving_envs_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 6.0),
      params={
        "velocity_range": PUSH_VELOCITY_RANGE,
        "command_name": TWIST_COMMAND_NAME,
        "command_threshold": 0.1,
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  """Velocity tracking plus light stabilization terms.

  ``upper_body_posture`` (10x below the tracking weights) keeps the arms
  near the nominal pose; ``self_collisions`` penalizes robot self-contacts.
  Gait style is otherwise left to the frozen HMP.
  """
  return {
    "track_linear_velocity": RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": TWIST_COMMAND_NAME, "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=vel_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": TWIST_COMMAND_NAME, "std": math.sqrt(0.5)},
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
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.5,
      params={"sensor_name": "self_collision"},
    ),
  }


def _terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=vel_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }


def _curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "command_vel": CurriculumTermCfg(
      func=vel_mdp.commands_vel,
      params={
        "command_name": TWIST_COMMAND_NAME,
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (0.0, 0.5),
            "lin_vel_y": (-0.08, 0.08),
            "ang_vel_z": (-0.5, 0.5),
          },
          {
            "step": 500 * 24,
            "lin_vel_x": (-0.2, 0.8),
            "lin_vel_y": (-0.2, 0.2),
            "ang_vel_z": (-1.0, 1.0),
          },
          {
            "step": 1000 * 24,
            "lin_vel_x": (-0.5, 1.5),
            "lin_vel_y": (-0.4, 0.4),
            "ang_vel_z": (-1.2, 1.2),
          },
          {
            "step": 2000 * 24,
            "lin_vel_x": (-0.8, 2.5),
            "lin_vel_y": (-0.6, 0.6),
            "ang_vel_z": (-2.0, 2.0),
          },
          {
            "step": 3000 * 24,
            "lin_vel_x": (-1.0, 3.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-3.0, 3.0),
          },
        ],
      },
    ),
  }


def make_locomotion_env_cfg() -> ManagerBasedRlEnvCfg:
  """Velocity-tracking template: actor obs ``[c | p | history]``."""
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

  observations = {
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

  commands: dict[str, CommandTermCfg] = {
    TWIST_COMMAND_NAME: UniformVelocityCommandCfg(**_velocity_command_kwargs()),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=_actions(),
    commands=commands,
    events=_events(),
    curriculum=_curriculum(),
    rewards=_rewards(),
    terminations=_terminations(),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=50,
      njmax=400,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
