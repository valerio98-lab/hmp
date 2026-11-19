import json

import numpy as np
import onnx
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction

from hmp.mdp.motion.base import MotionCommand


def _list_to_csv(arr, *, delimiter: str = ",") -> str:
  values: list[str] = []
  for x in arr:
    if isinstance(x, bool):
      values.append(str(x))
    elif isinstance(x, (int, np.integer)):
      values.append(str(int(x)))
    elif isinstance(x, (float, np.floating)):
      values.append(format(float(x), ".17g"))
    else:
      values.append(str(x))
  return delimiter.join(values)


def _export_value(value):
  if value is None:
    return None
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu()
    if value.ndim == 0:
      return value.item()
    if value.ndim > 1:
      value = value[0]
    return value.tolist()
  return value


def _serialize_metadata_value(value) -> str:
  if isinstance(value, torch.Tensor):
    value = _export_value(value)
  if isinstance(value, (list, tuple)):
    if all(isinstance(item, (str, int, float, bool)) for item in value):
      return _list_to_csv(list(value))
    return json.dumps(value)
  if isinstance(value, dict):
    return json.dumps(value)
  return str(value)


def _get_action_term_metadata(action_term) -> dict:
  if not isinstance(action_term, JointPositionAction):
    raise TypeError(
      f"Unsupported action term for export: {type(action_term).__name__}."
    )
  metadata = {
    "semantics": "joint_position",
    "dim": int(action_term.action_dim),
    "target_names": list(action_term.target_names),
    "scale": _export_value(action_term.scale),
    "offset": _export_value(action_term.offset),
  }
  return metadata


def _get_action_metadata(
  env: ManagerBasedRlEnv,
) -> dict[str, list | str | float | int | bool]:
  action_manager = env.action_manager
  if len(action_manager.active_terms) != 1:
    raise ValueError(
      "ONNX metadata export currently supports a single action term for deployment."
    )

  primary = _get_action_term_metadata(
    action_manager.get_term(action_manager.active_terms[0])
  )
  return {
    "action_semantics": primary["semantics"],
    "action_dim": primary["dim"],
    "action_target_names": primary["target_names"],
    "action_scale": primary["scale"],
    "action_offset": primary["offset"],
  }


def _get_observation_metadata(
  env: ManagerBasedRlEnv,
) -> dict[str, list | str | float | int | bool]:
  observation_manager = env.observation_manager
  group_name = "actor"
  observation_terms = observation_manager.active_terms[group_name]
  term_dims = observation_manager._group_obs_term_dim[group_name]
  term_cfgs = observation_manager._group_obs_term_cfgs[group_name]
  group_dim = observation_manager.group_obs_dim[group_name]

  terms_layout = []
  offset = 0
  for name, dims, term_cfg in zip(
    observation_terms, term_dims, term_cfgs, strict=False
  ):
    flat_dim = int(np.prod(dims))
    terms_layout.append(
      {
        "name": name,
        "shape": list(dims),
        "flat_dim": flat_dim,
        "slice": [offset, offset + flat_dim],
        "history_length": int(term_cfg.history_length),
        "flatten_history_dim": bool(term_cfg.flatten_history_dim),
      }
    )
    offset += flat_dim

  return {
    "observation_group": group_name,
    "observation_dim": list(group_dim) if isinstance(group_dim, tuple) else group_dim,
    "observation_terms_layout": terms_layout,
  }


def _get_motion_metadata(
  env: ManagerBasedRlEnv,
) -> dict[str, list | str | float | int | bool]:
  motion_term = env.command_manager.get_term("motion")
  assert isinstance(motion_term, MotionCommand)
  representation_name = "default"
  representation = motion_term.get_command_representation(representation_name)
  step_offsets = tuple(motion_term.future_sampling_step_offsets)

  return {
    "motion_command_class": type(motion_term).__name__,
    "motion_command_cfg_class": type(motion_term.cfg).__name__,
    "motion_command_representation_name": representation_name,
    "motion_command_dim": int(representation.shape[-1]),
    "motion_command_step_offsets": list(step_offsets),
    "anchor_body_name": motion_term.cfg.anchor_body_name,
    "body_names": list(getattr(motion_term.cfg, "body_names", ())),
    "root_body_name": motion_term.root_body_name,
  }


def _get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float | int | bool]:
  robot: Entity = env.scene["robot"]

  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names
    if jname in joint_name_to_ctrl_id
  ]

  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  metadata: dict[str, list | str | float | int | bool] = {
    "run_path": run_path,
    "physics_dt": float(env.physics_dt),
    "control_dt": float(env.step_dt),
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
  }
  metadata.update(_get_action_metadata(env))
  metadata.update(_get_observation_metadata(env))
  return metadata


def attach_onnx_metadata(env: ManagerBasedRlEnv, run_path: str, onnx_path: str) -> None:
  model = onnx.load(onnx_path)
  metadata = _get_base_metadata(env, run_path)
  if "motion" in env.command_manager.active_terms:
    metadata.update(_get_motion_metadata(env))

  existing = {entry.key: entry.value for entry in model.metadata_props}
  del model.metadata_props[:]
  for key, value in metadata.items():
    existing[key] = _serialize_metadata_value(value)

  for key, value in existing.items():
    entry = onnx.StringStringEntryProto()
    entry.key = key
    entry.value = value
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
