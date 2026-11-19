"""Unitree G1 environment configurations for the HMP pipeline."""

from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  HOME_KEYFRAME,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from hmp.hmp_env_cfg import make_env_cfg
from hmp.hmp_locomotion_env_cfg import make_locomotion_env_cfg
from hmp.hmp_navigation_env_cfg import make_navigation_env_cfg
from hmp.hmp_recovery_env_cfg import make_recovery_env_cfg
from hmp.mdp import JointRefAnchorRpMotionCommandCfg

DEFAULT_HMP_MOTION_SOURCE = str(
  Path(__file__).resolve().with_name("motion_data_cfg.yaml")
)
G1_HMP_TRACKED_BODY_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
_G1_FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"


def _apply_g1_common_overrides(cfg: ManagerBasedRlEnvCfg, robot_cfg) -> None:
  """Robot, contact sensors, action scale and foot-friction DR (all phases)."""
  # Self-collisions are enabled in every phase so the motion primitives are
  # learned, distilled and reused under the same contact physics.
  robot_cfg.collisions = (FULL_COLLISION,)
  cfg.scene.entities = {"robot": robot_cfg}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
  cfg.sim.contact_sensor_maxmatch = 500

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = dict(G1_ACTION_SCALE)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _G1_FOOT_GEOMS
  for observation_group in cfg.observations.values():
    if "friction_coeff" in observation_group.terms:
      observation_group.terms["friction_coeff"].params[
        "asset_cfg"
      ].geom_names = _G1_FOOT_GEOMS

  cfg.viewer.body_name = "pelvis"


def _disable_observation_noise(cfg: ManagerBasedRlEnvCfg) -> None:
  for observation_group in cfg.observations.values():
    observation_group.enable_corruption = False


def unitree_g1_hmp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Phase I/II motion-imitation environment on the G1."""
  cfg = make_env_cfg()
  _apply_g1_common_overrides(cfg, get_g1_robot_cfg())

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, JointRefAnchorRpMotionCommandCfg)
  motion_cmd.motion_file = DEFAULT_HMP_MOTION_SOURCE
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.root_body_name = "pelvis"
  motion_cmd.body_names = G1_HMP_TRACKED_BODY_NAMES
  motion_cmd.sampling_mode = "adaptive"

  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  if play:
    cfg.episode_length_s = int(1e9)
    _disable_observation_noise(cfg)
    # Keep only the end-of-clip termination; play clips from their start.
    motion_expired = cfg.terminations["motion_ref_expired"]
    cfg.terminations.clear()
    cfg.terminations["motion_ref_expired"] = motion_expired
    cfg.events.pop("push_robot")
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
  return cfg


def _apply_g1_task_overrides(
  cfg: ManagerBasedRlEnvCfg, play: bool
) -> ManagerBasedRlEnvCfg:
  """G1 overrides shared by the Phase III task environments."""
  robot_cfg = get_g1_robot_cfg()
  robot_cfg.init_state = HOME_KEYFRAME
  _apply_g1_common_overrides(cfg, robot_cfg)
  if play:
    cfg.episode_length_s = int(1e9)
    _disable_observation_noise(cfg)
    cfg.events.pop("push_robot")
  return cfg


def unitree_g1_hmp_locomotion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Phase III velocity tracking on the G1."""
  return _apply_g1_task_overrides(make_locomotion_env_cfg(), play=play)


def unitree_g1_hmp_navigation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Phase III point-goal navigation on the G1."""
  return _apply_g1_task_overrides(make_navigation_env_cfg(), play=play)


def unitree_g1_hmp_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Phase III fall-recovery velocity tracking on the G1."""
  cfg = _apply_g1_task_overrides(make_recovery_env_cfg(), play=play)
  if play:
    # Knockdowns from t=0 in play (the training gate would otherwise keep
    # them off for the first 3500 * 24 steps).
    cfg.events["knockdown_robot"].params["start_step"] = 0
  return cfg
