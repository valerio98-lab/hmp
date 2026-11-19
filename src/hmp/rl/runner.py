import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import wandb
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from hmp.rl.exporter import attach_onnx_metadata
from hmp.rl.locomotion_policy import HmpLocomotionActorModel
from hmp.rl.motion_stats import dump_motion_stats
from hmp.rl.reward_logging import (
  install_actual_episode_length_reward_logging,
  install_hmp_logger,
  install_merged_timeout_termination_logging,
)
from hmp.utils import get_wandb_checkpoint_path

# Shared history-encoder / latent defaults (paper Sec. IV-A: H = 10, all
# embeddings and code vectors are 128-dimensional).
_HISTORY_ENCODER_DEFAULTS: dict[str, Any] = {
  "history_latent_dim": 128,
  "history_conv_channels": (64, 32),
  "history_conv_kernel_sizes": (4, 2),
  "history_conv_strides": (2, 1),
  "layer_norm": True,
}
_PROPRIO_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)


def _flat_term_dims(observation_manager, group: str) -> dict[str, int]:
  terms = observation_manager.active_terms[group]
  dims = observation_manager._group_obs_term_dim[group]
  return {name: int(math.prod(d)) for name, d in zip(terms, dims, strict=True)}


def _history_steps(history_dim: int, step_dim: int, model_name: str) -> int:
  if step_dim <= 0 or history_dim % step_dim != 0:
    raise ValueError(
      f"{model_name} history dimension mismatch: "
      f"history_dim={history_dim}, history_step_dim={step_dim}."
    )
  return history_dim // step_dim


def configure_actor_obs_dims(
  observation_manager, group: str, model_name: str
) -> dict[str, int]:
  """Split an actor group laid out as ``[command | proprio | history]``."""
  flat = _flat_term_dims(observation_manager, group)
  command_dim = flat["command"]
  history_dim = flat["history"]
  proprio_dim = (
    int(observation_manager.group_obs_dim[group][0]) - command_dim - history_dim
  )
  return {
    "command_dim": command_dim,
    "proprio_dim": proprio_dim,
    "history_steps": _history_steps(history_dim, proprio_dim, model_name),
  }


def _configure_critic_obs_dims(
  observation_manager, group: str, model_name: str
) -> dict[str, int]:
  """Split a critic group laid out as
  ``[command | proprio | policy_history | privileged]``."""
  flat = _flat_term_dims(observation_manager, group)
  command_dim = flat["command"]
  history_dim = flat["policy_history"]
  proprio_dim = sum(flat[name] for name in _PROPRIO_TERMS)
  total = int(observation_manager.group_obs_dim[group][0])
  return {
    "command_dim": command_dim,
    "proprio_dim": proprio_dim,
    "privileged_dim": total - command_dim - proprio_dim - history_dim,
    "history_steps": _history_steps(history_dim, proprio_dim, model_name),
  }


class HmpOnPolicyRunner(MjlabOnPolicyRunner):
  """PPO runner for the Phase I expert (and base of the Phase III runner)."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    self._configure_model_cfg(env, train_cfg)
    install_actual_episode_length_reward_logging(env)
    install_merged_timeout_termination_logging(env)
    super().__init__(env, train_cfg, log_dir, device)
    install_hmp_logger(self)
    self.registry_name = registry_name

  def _maybe_upload_checkpoint(self, checkpoint_path: Path) -> None:
    if not self.cfg.get("upload_model", True):
      return
    mode = str(self.cfg.get("upload_model_mode", "rolling_latest"))
    if mode == "all":
      self.logger.save_model(str(checkpoint_path), self.current_learning_iteration)
      return
    if mode != "rolling_latest":
      raise ValueError(
        f"Unsupported upload_model_mode `{mode}`. Expected one of: all, rolling_latest."
      )
    latest_path = checkpoint_path.with_name("model_latest.pt")
    shutil.copy2(checkpoint_path, latest_path)
    self.logger.save_model(str(latest_path), self.current_learning_iteration)

  @staticmethod
  def _obs_group_name(train_cfg: dict, obs_set: str) -> str:
    group_names = train_cfg.get("obs_groups", {}).get(obs_set, (obs_set,))
    if isinstance(group_names, str):
      return group_names
    return str(group_names[0]) if group_names else obs_set

  @classmethod
  def _configure_model_cfg(cls, env: RslRlVecEnvWrapper, train_cfg: dict) -> None:
    """Fill model kwargs (obs split, history length) from the env layout."""
    actor_cfg = train_cfg["actor"]
    critic_cfg = train_cfg["critic"]
    actor_class = str(actor_cfg.get("class_name", ""))
    critic_class = str(critic_cfg.get("class_name", ""))
    obs_manager = env.unwrapped.observation_manager
    actor_group = cls._obs_group_name(train_cfg, "actor")
    critic_group = cls._obs_group_name(train_cfg, "critic")

    if actor_class.endswith(":HmpEncoderDecoderActorModel"):
      dims = configure_actor_obs_dims(obs_manager, actor_group, actor_class)
      actor_cfg.setdefault("current_motion_obs_dim", dims["command_dim"])
      actor_cfg.setdefault("proprio_obs_dim", dims["proprio_dim"])
      actor_cfg.setdefault("history_steps", dims["history_steps"])
      for key, value in _HISTORY_ENCODER_DEFAULTS.items():
        actor_cfg.setdefault(key, value)
      actor_cfg.setdefault("latent_dim", 128)
      actor_cfg.setdefault("encoder_hidden_dims", (512, 512, 256, 128))

    elif actor_class.endswith(":HmpLocomotionActorModel"):
      dims = configure_actor_obs_dims(obs_manager, actor_group, actor_class)
      actor_cfg.setdefault("task_goal_obs_dim", dims["command_dim"])
      actor_cfg.setdefault("proprio_obs_dim", dims["proprio_dim"])
      actor_cfg.setdefault("history_steps", dims["history_steps"])
      for key, value in _HISTORY_ENCODER_DEFAULTS.items():
        actor_cfg.setdefault(key, value)
      actor_cfg.setdefault("latent_dim", 128)
      actor_cfg.setdefault("high_level_hidden_dims", (512, 512, 256, 128))
      actor_cfg.setdefault(
        "rvq_num_active_quantizers", train_cfg.get("rvq_num_active_quantizers")
      )

    else:
      raise ValueError(f"Unsupported actor class: {actor_class!r}")

    if critic_class.endswith((":HmpEncoderDecoderCriticModel", ":HmpCriticModel")):
      dims = _configure_critic_obs_dims(obs_manager, critic_group, critic_class)
      critic_cfg.setdefault("current_motion_obs_dim", dims["command_dim"])
      critic_cfg.setdefault("proprio_obs_dim", dims["proprio_dim"])
      critic_cfg.setdefault("privileged_obs_dim", dims["privileged_dim"])
      critic_cfg.setdefault("history_steps", dims["history_steps"])
      for key, value in _HISTORY_ENCODER_DEFAULTS.items():
        critic_cfg.setdefault(key, value)
      if critic_class.endswith(":HmpEncoderDecoderCriticModel"):
        critic_cfg.setdefault("latent_dim", 128)
        critic_cfg.setdefault("encoder_hidden_dims", (512, 512, 256, 128))
    else:
      raise ValueError(f"Unsupported critic class: {critic_class!r}")

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    self._maybe_upload_checkpoint(Path(path))
    dump_motion_stats(
      self.env,
      self.cfg,
      Path(path),
      self.current_learning_iteration,
      logger_type=getattr(self.logger, "logger_type", None),
    )

    policy_path = Path(path).parent
    filename = f"{policy_path.parent.name}.onnx"
    try:
      self.export_policy_to_onnx(str(policy_path), filename)
      run_name = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )
      attach_onnx_metadata(self.env.unwrapped, run_name, str(policy_path / filename))
      if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
        wandb.save(
          str(policy_path / filename),
          base_path=os.path.dirname(str(policy_path)),
        )
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore[union-attr]
          self.registry_name = None
    except Exception as exc:
      print(f"[WARN] ONNX export failed (training continues): {exc}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if load_iteration and infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos


class HmpLocomotionOnPolicyRunner(HmpOnPolicyRunner):
  """Phase III runner: PPO over code selection on the frozen HMP.

  Loads a trained ``HmpImitationModel`` (HMP) checkpoint into the actor and
  freezes its history encoder, proprioceptive encoder, RVQ codebook and action
  decoder before PPO starts. Only the task-level policy is trained.
  """

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    self._hmp_log_dir = Path(log_dir).resolve() if log_dir is not None else None
    super().__init__(env, train_cfg, log_dir, device, registry_name=registry_name)
    self._load_imitation_checkpoint()
    self._assert_freezing_invariants()

  def _imitation_checkpoint_path(self) -> Path:
    checkpoint_file = self.cfg.get("imitation_checkpoint_file")
    if checkpoint_file:
      checkpoint_path = Path(str(checkpoint_file)).expanduser().resolve()
      if not checkpoint_path.exists():
        raise FileNotFoundError(
          f"Imitation checkpoint file not found: {checkpoint_path}"
        )
      print(f"[INFO]: Using local imitation checkpoint: {checkpoint_path}")
      return checkpoint_path

    wandb_run_path = self.cfg.get("imitation_wandb_run_path")
    if not wandb_run_path:
      raise ValueError(
        "HmpLocomotionOnPolicyRunner requires an HMP (imitation) "
        "checkpoint. Set --agent.imitation-checkpoint-file or "
        "--agent.imitation-wandb-run-path."
      )
    if self._hmp_log_dir is None:
      raise ValueError(
        "Cannot resolve imitation W&B checkpoint without a log directory."
      )
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      self._hmp_log_dir.parent,
      Path(str(wandb_run_path)),
      self.cfg.get("imitation_wandb_checkpoint_name"),
    )
    print(
      "[INFO]: Resolved imitation checkpoint: "
      f"{checkpoint_path.name} (run: {checkpoint_path.parent.name}, "
      f"{'cached' if was_cached else 'downloaded'})"
    )
    return checkpoint_path

  def _load_imitation_checkpoint(self) -> None:
    loaded = torch.load(
      self._imitation_checkpoint_path(),
      map_location=self.device,
      weights_only=False,
    )
    trainer_blob = loaded.get("trainer")
    if not isinstance(trainer_blob, dict) or "model" not in trainer_blob:
      raise ValueError(
        "Not an HMP checkpoint: expected the `trainer.model` state dict "
        "written by HmpImitationRunner."
      )
    actor = self.alg.actor
    assert isinstance(actor, HmpLocomotionActorModel)
    actor.load_imitation_weights(trainer_blob["model"])
    actor.train(actor.training)

  def _assert_freezing_invariants(self) -> None:
    actor = self.alg.actor
    assert isinstance(actor, HmpLocomotionActorModel)
    frozen_names = set(actor.FROZEN_SUBMODULES) | {"rvq"}
    trainable_names = {
      name.split(".", 1)[0]
      for name, param in actor.named_parameters()
      if param.requires_grad
    }
    forbidden = trainable_names & frozen_names
    if forbidden:
      raise RuntimeError(
        "Locomotion actor has trainable parameters in frozen submodules: "
        f"{sorted(forbidden)}."
      )
    print(
      "[INFO]: Locomotion actor trainable top-level submodules: "
      f"{sorted(trainable_names)}; frozen: {sorted(frozen_names)}."
    )
