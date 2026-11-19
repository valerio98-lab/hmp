"""Runner configurations for the three-stage HMP pipeline on the Unitree G1."""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from hmp.rl import (
  HmpImitationRunnerCfg,
  HmpLocomotionOnPolicyRunnerCfg,
  HmpOnPolicyRunnerCfg,
  ImitationLossWeights,
  ImitationTrainerCfg,
)

_HIDDEN_DIMS = (512, 512, 256, 128)
_OBS_GROUPS = {"actor": ("actor",), "critic": ("critic",)}


def _expert_actor_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    class_name="hmp.rl.policy:HmpEncoderDecoderActorModel",
    hidden_dims=_HIDDEN_DIMS,
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "log",
    },
  )


def unitree_g1_hmp_encdec_ppo_runner_cfg() -> HmpOnPolicyRunnerCfg:
  """Phase I: motion-imitation expert (PPO, asymmetric actor-critic, 30k iters)."""
  return HmpOnPolicyRunnerCfg(
    seed=1,
    actor=_expert_actor_cfg(),
    critic=RslRlModelCfg(
      class_name="hmp.rl.policy:HmpEncoderDecoderCriticModel",
      hidden_dims=_HIDDEN_DIMS,
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hmp_encdec",
    wandb_project="hmp",
    wandb_tags=("hmp", "expert", "encoder_decoder"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    obs_groups=_OBS_GROUPS,
  )


def unitree_g1_hmp_imitation_runner_cfg() -> HmpImitationRunnerCfg:
  """Phase II: HMP distillation (lambda_a=10, lambda_m=0.1->1, lambda_r=0.05,
  lambda_q=1; 6k iters)."""
  return HmpImitationRunnerCfg(
    seed=1,
    expert=_expert_actor_cfg(),
    student=RslRlModelCfg(
      class_name="hmp.rl.imitation_RVQ_policy:HmpImitationModel",
      hidden_dims=_HIDDEN_DIMS,
      activation="elu",
      obs_normalization=True,
    ),
    loss_weights=ImitationLossWeights(action=10.0, mm=1.0, reg=0.05, vq=1.0),
    trainer=ImitationTrainerCfg(
      lr=2e-4,
      grad_clip_norm=1.0,
      mm_warmup_steps=10_000,
      mm_start=0.1,
      mm_end=1.0,
    ),
    experiment_name="g1_hmp_imitation",
    wandb_project="hmp",
    wandb_tags=("hmp", "imitation", "rvq"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=6_000,
    obs_groups=_OBS_GROUPS,
  )


def unitree_g1_hmp_locomotion_runner_cfg() -> HmpLocomotionOnPolicyRunnerCfg:
  """Phase III: velocity tracking by code selection on the frozen HMP."""
  return HmpLocomotionOnPolicyRunnerCfg(
    seed=1,
    # Active RVQ depth M_act (Sec. V-B); None uses all 8 stages. Override from
    # the CLI, e.g. `--agent.rvq-num-active-quantizers 3`.
    rvq_num_active_quantizers=None,
    actor=RslRlModelCfg(
      class_name="hmp.rl.locomotion_policy:HmpLocomotionActorModel",
      hidden_dims=_HIDDEN_DIMS,
      activation="elu",
      obs_normalization=True,
      distribution_cfg=None,
    ),
    critic=RslRlModelCfg(
      class_name="hmp.rl.policy:HmpCriticModel",
      hidden_dims=_HIDDEN_DIMS,
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="hmp.rl.categorical_ppo:HmpCategoricalPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Categorical entropy is ~55 nat at uniform (8 heads x log(1024)):
      # 0.01 made the bonus dominate the surrogate ~9x and pinned the
      # policy near uniform; 0.002 keeps exploration without freezing it.
      entropy_coef=0.002,
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_hmp_locomotion",
    wandb_project="hmp",
    wandb_tags=("hmp", "locomotion", "frozen_hmp", "high_level"),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=4_500,
    obs_groups=_OBS_GROUPS,
  )


def unitree_g1_hmp_navigation_runner_cfg() -> HmpLocomotionOnPolicyRunnerCfg:
  """Phase III: point-goal navigation. Same setup as velocity tracking; the
  3-D goal command dims are derived by the runner from the env."""
  cfg = unitree_g1_hmp_locomotion_runner_cfg()
  cfg.experiment_name = "g1_hmp_navigation"
  cfg.wandb_tags = ("hmp", "navigation", "frozen_hmp", "high_level")
  cfg.max_iterations = 5_500
  return cfg


def unitree_g1_hmp_recovery_runner_cfg() -> HmpLocomotionOnPolicyRunnerCfg:
  """Phase III: fall-recovery velocity tracking. Same setup and 3-D twist
  command as velocity tracking. The HMP must be distilled from a motion set
  that includes the fallAndGetUp clips"""
  cfg = unitree_g1_hmp_locomotion_runner_cfg()
  cfg.experiment_name = "g1_hmp_recovery"
  cfg.wandb_tags = ("hmp", "recovery", "frozen_hmp", "high_level")
  cfg.max_iterations = 9_000
  return cfg
