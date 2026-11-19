from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.utils import resolve_callable

from hmp.rl.imitation_trainer import (
  ImitationLossWeights,
  ImitationTrainer,
  ImitationTrainerCfg,
)
from hmp.rl.runner import _HISTORY_ENCODER_DEFAULTS, configure_actor_obs_dims


class HmpImitationRunner:
  """Phase II runner: distill the frozen expert into `HmpImitationModel`.

  Each iteration:
    1. Roll out the frozen encoder-decoder expert in the env for
       ``num_steps_per_env`` steps, recording (obs, expert_action) at each step.
       The expert's deterministic action is used to step the env, so the
       collected state distribution is the expert's on-policy distribution.
    2. Update the `HmpImitationModel` on the collected (obs, expert_action)
       batch using `ImitationTrainer`. No PPO, no value function — pure
       supervised imitation with the composite loss of Eq. (10).

  The runner expects the following keys in ``train_cfg``:
    - ``expert``: dict with ``class_name`` plus model kwargs for the expert.
    - ``student``: dict with ``class_name`` plus model kwargs for the student.
    - ``expert_checkpoint``: path to the trained encoder-decoder checkpoint.
    - ``loss_weights``: kwargs for `ImitationLossWeights`.
    - ``trainer``: kwargs for `ImitationTrainerCfg`.
    - ``num_steps_per_env``: rollout length per iteration.
    - ``save_interval``: iterations between checkpoint saves.
    - ``obs_groups``: dict matching the on-policy runner format.
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: Any,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cuda",
    **kwargs: Any,
  ) -> None:
    del kwargs  # absorb mjlab.scripts.train extras like `registry_name`
    if not isinstance(env, RslRlVecEnvWrapper):
      raise TypeError("HmpImitationRunner requires an RslRlVecEnvWrapper-wrapped env.")
    self.env = env
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir

    self._configure_model_cfg(env, train_cfg)

    obs = env.get_observations()
    if isinstance(obs, tuple):
      obs = obs[0]

    obs_groups = train_cfg.get("obs_groups", {"actor": ("actor",)})
    action_dim = int(env.num_actions)

    expert_cfg = dict(train_cfg["expert"])
    expert_cls = resolve_callable(expert_cfg.pop("class_name"))
    expert_cfg.pop(
      "cnn_cfg", None
    )  # expert doesn't use the RSL private CNN, so ignore any cnn_cfg
    self.expert = expert_cls(
      obs=obs,
      obs_groups=obs_groups,
      obs_set="actor",
      output_dim=action_dim,
      **expert_cfg,
    ).to(device)

    # Required for training (checked in `learn`); `play` only needs the
    # student, whose weights come from the distilled checkpoint.
    self._expert_loaded = False
    expert_ckpt = train_cfg.get("expert_checkpoint")
    if expert_ckpt is not None:
      self._load_expert(expert_ckpt)
    self.expert.eval()
    for p in self.expert.parameters():
      p.requires_grad = False

    student_cfg = dict(train_cfg["student"])
    student_cls = resolve_callable(student_cfg.pop("class_name"))
    student_cfg.pop(
      "cnn_cfg", None
    )  # student doesn't use the RSL private CNN, so ignore any cnn_cfg
    self.student = student_cls(
      obs=obs,
      obs_groups=obs_groups,
      obs_set="actor",
      output_dim=action_dim,
      **student_cfg,
    ).to(device)

    loss_weights = ImitationLossWeights(**train_cfg.get("loss_weights", {}))
    trainer_cfg = ImitationTrainerCfg(**train_cfg.get("trainer", {}))
    self.trainer = ImitationTrainer(
      model=self.student,
      loss_weights=loss_weights,
      cfg=trainer_cfg,
      device=device,
    )

    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.save_interval = int(train_cfg.get("save_interval", 500))
    self.current_iteration = 0

    self._wandb_initialized = False
    self._use_wandb = (
      train_cfg.get("logger", "wandb") == "wandb" and log_dir is not None
    )

    # The student starts from the expert's fitted obs-normalizer, so its
    # modules see normalized inputs from the first update. Phase III later
    # copies this normalizer into the task policy.
    if self._expert_loaded:
      self._warmstart_student_normalizer_from_expert()

  def _warmstart_student_normalizer_from_expert(self) -> None:
    """Copy the expert's fitted obs-normalizer (mean/var/std/count) into
    the student. Expert and student share the observation layout."""
    if not (self.student.obs_normalization and self.expert.obs_normalization):
      return
    expert_norm = self.expert.obs_normalizer
    student_norm = self.student.obs_normalizer
    with torch.no_grad():
      student_norm._mean.copy_(expert_norm._mean)
      student_norm._var.copy_(expert_norm._var)
      student_norm._std.copy_(expert_norm._std)
      student_norm.count.copy_(expert_norm.count)
    print(
      "[HmpImitationRunner] Warm-started student obs-normalizer from "
      f"expert (count={int(student_norm.count.item())})."
    )

  def _load_expert(self, ckpt_path: str | os.PathLike) -> None:
    print(f"[HmpImitationRunner] Loading expert checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
    if "actor_state_dict" not in ckpt:
      raise ValueError(
        f"{ckpt_path} is not a Phase I expert checkpoint (missing `actor_state_dict`)."
      )
    # Strict: a silently partial load would distill a random expert.
    self.expert.load_state_dict(ckpt["actor_state_dict"], strict=True)
    self._expert_loaded = True

  @torch.no_grad()
  def _expert_action(self, obs) -> torch.Tensor:
    """Deterministic mean action from the expert."""
    return self.expert(obs)

  def add_git_repo_to_log(self, *args, **kwargs) -> None:
    pass

  def _maybe_init_wandb(self) -> None:
    if not self._use_wandb or self._wandb_initialized:
      return
    try:
      import wandb
    except ImportError:
      print("[HmpImitationRunner] wandb not installed — skipping logging.")
      self._use_wandb = False
      return
    entity = os.environ.get("WANDB_USERNAME") or os.environ.get("WANDB_ENTITY")
    wandb.init(
      project=self.cfg.get("wandb_project", "hmp"),
      entity=entity,
      name=Path(self.log_dir).name if self.log_dir else None,
      tags=list(self.cfg.get("wandb_tags", ())),
      config=self.cfg,
    )
    self._wandb_initialized = True

  def learn(self, num_learning_iterations: int, **kwargs: Any) -> None:
    del kwargs  # absorb `init_at_random_ep_len` etc.
    if not self._expert_loaded:
      raise ValueError(
        "HMP distillation requires the Phase I expert: pass "
        "--agent.expert-checkpoint <path/to/model.pt>."
      )
    self._maybe_init_wandb()
    obs = self.env.get_observations()
    if isinstance(obs, tuple):
      obs = obs[0]
    obs = obs.to(self.device)

    num_envs = int(self.env.num_envs)
    cur_ep_len = torch.zeros(num_envs, device=self.device, dtype=torch.long)

    target_iteration = self.current_iteration + int(num_learning_iterations)
    while self.current_iteration < target_iteration:
      rollout_obs: list = []
      rollout_actions: list[torch.Tensor] = []
      rollout_dones: list[torch.Tensor] = []
      ended_lengths: list[torch.Tensor] = []

      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          action = self._expert_action(obs)
          rollout_obs.append(obs.clone())
          rollout_actions.append(action.clone())
          obs, _, dones, _ = self.env.step(action.to(self.env.device))
          dones = dones.to(self.device)
          rollout_dones.append(dones.clone())
          cur_ep_len += 1
          done_mask = dones.bool()
          if done_mask.any():
            ended_lengths.append(cur_ep_len[done_mask].clone())
            cur_ep_len[done_mask] = 0
          obs = obs.to(self.device)
          # Standard running update of the student's normalizer,
          # starting from the expert statistics.
          self.student.update_normalization(obs)

      stats: dict[str, float] = {}
      for i, (step_obs, step_action) in enumerate(
        zip(rollout_obs, rollout_actions, strict=True)
      ):
        step_obs = step_obs.clone()
        step_action = step_action.clone()
        batch: dict = {"obs": step_obs, "a_expert": step_action}
        if i > 0:
          batch["prev_done"] = rollout_dones[i - 1]
        stats = self.trainer.train_step(batch)

      if ended_lengths:
        lens = torch.cat(ended_lengths).float()
        stats["expert/episode_length_mean"] = float(lens.mean().item())
        stats["expert/episode_length_max"] = float(lens.max().item())
        stats["expert/num_episodes"] = float(lens.numel())
      else:
        stats["expert/episode_length_mean"] = float(cur_ep_len.float().mean().item())
        stats["expert/episode_length_max"] = float(cur_ep_len.max().item())
        stats["expert/num_episodes"] = 0.0

      self.current_iteration += 1
      it = self.current_iteration
      print(
        f"[iter {it}] "
        f"loss_total={stats.get('loss_total', float('nan')):.4f} "
        f"loss_action={stats.get('loss_action', float('nan')):.4f} "
        f"loss_mm={stats.get('loss_mm', float('nan')):.4f} "
        f"loss_commit={stats.get('loss_commit', float('nan')):.4f}"
      )

      if self._use_wandb and self._wandb_initialized:
        import wandb

        wandb.log(stats, step=it)

      if self.log_dir and it % self.save_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    if self.log_dir:
      self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))

  def save(self, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "trainer": self.trainer.state_dict(),
        "iteration": self.current_iteration,
      },
      path,
    )
    if self._use_wandb and self._wandb_initialized:
      import wandb

      wandb.save(path, base_path=str(Path(path).parent), policy="now")

  def load(self, path: str, strict: bool = True, **kwargs: Any) -> None:
    del kwargs  # absorb mjlab.scripts.play extras like `load_cfg`, `map_location`
    sd = torch.load(path, map_location=self.device, weights_only=False)
    self.trainer.load_state_dict(sd["trainer"], strict=strict)
    self.current_iteration = int(sd.get("iteration", 0))

  def get_inference_policy(self, device: str | torch.device | None = None):
    """Return a callable mapping obs -> action for use in `play`."""
    if device is not None:
      self.student.to(device)
    self.student.eval()

    @torch.inference_mode()
    def policy(obs):
      out = self.student(obs)
      return out["a_hat"] if isinstance(out, dict) else out

    return policy

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the distilled tracker (reference encoder + HMP) to ONNX."""
    onnx_model = self.student.as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,
      output_names=onnx_model.output_names,
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _configure_model_cfg(env: RslRlVecEnvWrapper, train_cfg: dict) -> None:
    """Inject env-derived dims into the ``expert`` and ``student`` configs."""
    dims = configure_actor_obs_dims(
      env.unwrapped.observation_manager, "actor", "HmpImitationRunner"
    )
    for key in ("expert", "student"):
      cfg = train_cfg[key]
      cfg.setdefault("current_motion_obs_dim", dims["command_dim"])
      cfg.setdefault("proprio_obs_dim", dims["proprio_dim"])
      cfg.setdefault("history_steps", dims["history_steps"])
      for name, value in _HISTORY_ENCODER_DEFAULTS.items():
        cfg.setdefault(name, value)
      cfg.setdefault("latent_dim", 128)
    train_cfg["expert"].setdefault("encoder_hidden_dims", (512, 512, 256, 128))
    train_cfg["student"].setdefault(
      "rvq_rotation_trick", train_cfg.get("rvq_rotation_trick", True)
    )
