from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg, RslRlOnPolicyRunnerCfg

from hmp.rl.imitation_trainer import ImitationLossWeights, ImitationTrainerCfg

UploadModelMode = Literal["all", "rolling_latest"]


@dataclass
class HmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  upload_model_mode: UploadModelMode = "rolling_latest"


@dataclass
class HmpLocomotionOnPolicyRunnerCfg(HmpOnPolicyRunnerCfg):
  """Phase III PPO runner config (code selection on the frozen HMP)."""

  # HMP checkpoint from Phase II: a local file or a W&B run.
  imitation_checkpoint_file: str | None = None
  imitation_wandb_run_path: str | None = None
  imitation_wandb_checkpoint_name: str | None = None

  # Active RVQ depth M_act driven by the task policy (Sec. V-B). The frozen
  # RVQ keeps all its stages; None uses all of them.
  rvq_num_active_quantizers: int | None = None


@dataclass
class HmpImitationRunnerCfg(RslRlBaseRunnerCfg):
  """Config for the Phase II HMP distillation runner.

  Inherits the standard mjlab base runner fields (seed, max_iterations,
  experiment_name, clip_actions, …) so it plugs into ``mjlab.scripts.train``
  the same way the PPO runner does. The runner-specific fields below describe
  the frozen expert, the student model, and the supervised loss / optimizer.

  ``expert_checkpoint`` must be provided (CLI: ``--agent.expert-checkpoint``);
  it points to the trained encoder-decoder policy that supplies target
  actions during rollouts.
  """

  class_name: str = "hmp.rl.imitation_runner:HmpImitationRunner"
  expert: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="hmp.rl.policy:HmpEncoderDecoderActorModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    )
  )
  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="hmp.rl.imitation_RVQ_policy:HmpImitationModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    )
  )
  expert_checkpoint: str | None = None
  loss_weights: ImitationLossWeights = field(default_factory=ImitationLossWeights)
  trainer: ImitationTrainerCfg = field(default_factory=ImitationTrainerCfg)
  # Codebook training with the rotation trick (True, paper default) or with
  # the standard straight-through estimator (False), Sec. V-C.
  rvq_rotation_trick: bool = True
