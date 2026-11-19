"""Phase III actor: task-level code-selection policy on top of the frozen HMP.

Forward pipeline (Sec. III-C):

    1. obs_normalizer(obs) -> [c | p | history]   (c = task command)
    2. eta    = H_alpha(history)                   (frozen)
    3. s      = [p, eta]
    4. logits = F_xi(s, c)  in R^{M_act x K}       (trained), Eq. (18)
    5. k^(m)  ~ Cat(softmax(logits^(m)))           (argmax at evaluation), Eq. (19)
    6. y_hat  = sum_m e^(m)_{k^(m)}                (frozen codebook), Eq. (7)
    7. a      = D_psi(s, P_theta(s) + y_hat)       (frozen), Eq. (17)

PPO stores the sampled indices (``HmpCategoricalPPO``); the environment is
stepped with the continuous joint-position action produced by the frozen HMP.

Observation normalizer: the ``[p | history]`` slots (and the sample ``count``)
are copied from the HMP checkpoint, so the frozen modules see the input
statistics they were distilled with. The command slot ``c`` is initialized to
identity so the command enters in physical units. The normalizer keeps the
standard rsl_rl running update during PPO; the inherited count (~3.5e9 samples
for the paper HMP) weights the Phase III samples accordingly.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from hmp.rl.distributions import MultiCategoricalDistribution
from hmp.rl.imitation_RVQ_policy import _ActionDecoder, _PriorNet
from hmp.rl.policy import MotionEncoder, _build_mlp
from hmp.rl.residual_vq import ResidualVQ, RVQCfg


class _CategoricalHighLevel(nn.Module):
  """Task-level policy F_xi: ``(s, c)`` -> flat logits ``(B, M_act * K)``."""

  def __init__(
    self,
    s_dim: int,
    goal_dim: int,
    num_heads: int,
    codebook_size: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    layer_norm: bool,
  ) -> None:
    super().__init__()
    self.net = _build_mlp(
      input_dim=int(s_dim) + int(goal_dim),
      output_dim=int(num_heads) * int(codebook_size),
      hidden_dims=hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )

  def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat((s, goal), dim=-1))


class HmpLocomotionActorModel(MLPModel):
  """Frozen HMP + trainable categorical code-selection policy."""

  # HMP modules loaded from the Phase II checkpoint and kept frozen.
  FROZEN_SUBMODULES: tuple[str, ...] = (
    "history_encoder",
    "prior",
    "rvq",
    "action_decoder",
  )

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    task_goal_obs_dim: int = 0,
    proprio_obs_dim: int = 0,
    history_steps: int = 10,
    history_latent_dim: int = 128,
    history_conv_channels: tuple[int, ...] | list[int] = (64, 32),
    history_conv_kernel_sizes: tuple[int, ...] | list[int] = (4, 2),
    history_conv_strides: tuple[int, ...] | list[int] = (2, 1),
    latent_dim: int = 128,
    high_level_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    layer_norm: bool = True,
    rvq_num_quantizers: int = 8,
    # Active RVQ depth M_act (Sec. V-B): the categorical drives only the
    # first M_act stages (coarse to fine); None uses all of them.
    rvq_num_active_quantizers: int | None = None,
    rvq_codebook_size: int = 1024,
  ) -> None:
    del distribution_cfg  # always the multi-categorical below.

    self.task_goal_obs_dim = int(task_goal_obs_dim)
    self.proprio_obs_dim = int(proprio_obs_dim)
    self.history_steps = int(history_steps)
    self.history_latent_dim = int(history_latent_dim)
    self.latent_dim = int(latent_dim)
    self.action_dim = int(output_dim)
    self.num_quantizers = int(rvq_num_quantizers)
    self.num_active_codebooks = (
      self.num_quantizers
      if rvq_num_active_quantizers is None
      else int(rvq_num_active_quantizers)
    )
    self.codebook_size = int(rvq_codebook_size)

    for name in (
      "task_goal_obs_dim",
      "proprio_obs_dim",
      "history_steps",
      "latent_dim",
      "num_quantizers",
      "codebook_size",
    ):
      if getattr(self, name) <= 0:
        raise ValueError(f"`{name}` must be positive, got {getattr(self, name)}.")
    if not 1 <= self.num_active_codebooks <= self.num_quantizers:
      raise ValueError(
        "`rvq_num_active_quantizers` must be in "
        f"[1, {self.num_quantizers}], got {self.num_active_codebooks}."
      )

    self.current_obs_dim = self.task_goal_obs_dim + self.proprio_obs_dim
    self.history_obs_dim = self.proprio_obs_dim * self.history_steps

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=None,
    )
    # The high-level net produces the distribution input directly.
    self.mlp = nn.Identity()
    self.distribution = MultiCategoricalDistribution(
      output_dim=self.num_active_codebooks,
      codebook_size=self.codebook_size,
    )

    expected_obs_dim = self.current_obs_dim + self.history_obs_dim
    if self.obs_dim != expected_obs_dim:
      raise ValueError(
        "HmpLocomotionActorModel observation dimension mismatch: "
        f"got {self.obs_dim}, expected {expected_obs_dim} "
        f"({self.current_obs_dim} current + {self.history_obs_dim} history)."
      )

    self.history_encoder = MotionEncoder(
      input_dim_per_step=self.proprio_obs_dim,
      num_steps=self.history_steps,
      activation=activation,
      conv_channels=history_conv_channels,
      conv_kernel_sizes=history_conv_kernel_sizes,
      conv_strides=history_conv_strides,
      projection_dim=self.history_latent_dim,
    )
    s_dim = self.proprio_obs_dim + self.history_latent_dim
    self.prior = _PriorNet(
      s_dim=s_dim,
      latent_dim=self.latent_dim,
      hidden_dims=high_level_hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )
    self.high_level = _CategoricalHighLevel(
      s_dim=s_dim,
      goal_dim=self.task_goal_obs_dim,
      num_heads=self.num_active_codebooks,
      codebook_size=self.codebook_size,
      hidden_dims=high_level_hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )
    self.action_decoder = _ActionDecoder(
      s_dim=s_dim,
      latent_dim=self.latent_dim,
      action_dim=self.action_dim,
      hidden_dims=hidden_dims,
      activation=activation,
    )
    # Lookup-only codebook container (structure must match the HMP).
    self.rvq = ResidualVQ(
      RVQCfg(
        dim=self.latent_dim,
        num_quantizers=self.num_quantizers,
        codebook_size=self.codebook_size,
        quantize_dropout=False,
      )
    )
    self._pin_gtask_normalizer_identity()

  def _get_latent_dim(self) -> int:
    # Only sizes the MLP that is replaced by nn.Identity above.
    return self.action_dim

  def _build_s_rich(self, obs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(s, c)`` with ``s = [p, eta]`` from a normalized obs vector."""
    g_task = obs_flat[:, : self.task_goal_obs_dim]
    proprio = obs_flat[:, self.task_goal_obs_dim : self.current_obs_dim]
    history_obs = obs_flat[
      :, self.current_obs_dim : self.current_obs_dim + self.history_obs_dim
    ]
    s_rich = torch.cat((proprio, self.history_encoder(history_obs)), dim=-1)
    return s_rich, g_task

  def _decode(self, s_rich: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Frozen HMP, Eq. (17): a = D(s, P(s) + sum_m e^(m)_{k^(m)})."""
    z_hat = self.prior(s_rich) + self.rvq.lookup(indices)
    return self.action_decoder(s_rich, z_hat)

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    """Return flat categorical logits ``(B, M_act * K)``."""
    obs_flat = super().get_latent(obs, masks, hidden_state)
    s_rich, g_task = self._build_s_rich(obs_flat)
    return self.high_level(s_rich, g_task)

  def indices_to_continuous_action(
    self, obs: TensorDict, indices: torch.Tensor
  ) -> torch.Tensor:
    """Decode sampled code indices into the joint-position action."""
    obs_flat = MLPModel.get_latent(self, obs, None, None)
    s_rich, _ = self._build_s_rich(obs_flat)
    return self._decode(s_rich, indices.to(obs_flat.device))

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    """Stochastic: sampled indices (B, M_act) for the PPO rollout.
    Deterministic: argmax indices decoded to the joint action (B, 29)."""
    if stochastic_output:
      return super().forward(obs, masks, hidden_state, stochastic_output=True)
    obs_flat = MLPModel.get_latent(self, obs, masks, hidden_state)
    s_rich, g_task = self._build_s_rich(obs_flat)
    logits = self.high_level(s_rich, g_task).view(
      -1, self.num_active_codebooks, self.codebook_size
    )
    return self._decode(s_rich, logits.argmax(dim=-1))

  def train(self, mode: bool = True) -> "HmpLocomotionActorModel":
    super().train(mode)
    for name in self.FROZEN_SUBMODULES:
      getattr(self, name).eval()
    return self

  def load_imitation_weights(
    self, imitation_state_dict: dict[str, torch.Tensor]
  ) -> None:
    """Load and freeze the HMP modules; copy the HMP normalizer statistics."""
    by_prefix: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in imitation_state_dict.items():
      prefix, _, sub_key = key.partition(".")
      if sub_key:
        by_prefix.setdefault(prefix, {})[sub_key] = tensor

    missing = [name for name in self.FROZEN_SUBMODULES if name not in by_prefix]
    if missing:
      raise KeyError(
        f"HMP checkpoint is missing submodules {missing}; expected "
        f"prefixes {list(self.FROZEN_SUBMODULES)}."
      )
    for name in self.FROZEN_SUBMODULES:
      module = getattr(self, name)
      module.load_state_dict(by_prefix[name], strict=True)
      module.eval()
      for p in module.parameters():
        p.requires_grad = False

    if self.obs_normalization:
      self._copy_normalizer_proprio_history(imitation_state_dict)
      self._pin_gtask_normalizer_identity()

  def _pin_gtask_normalizer_identity(self) -> None:
    """Set the command slot of the normalizer to mean 0, var 1, std 1."""
    if not self.obs_normalization:
      return
    end = self.task_goal_obs_dim
    with torch.no_grad():
      self.obs_normalizer._mean[:, :end].zero_()  # type: ignore[attr-defined]
      self.obs_normalizer._var[:, :end].fill_(1.0)  # type: ignore[attr-defined]
      self.obs_normalizer._std[:, :end].fill_(1.0)  # type: ignore[attr-defined]

  def _copy_normalizer_proprio_history(
    self, source_state_dict: dict[str, torch.Tensor]
  ) -> None:
    """Copy the ``[p | history]`` normalizer slots and the sample count.

    The HMP normalizer has layout ``[m (M) | p (P) | history (P*H)]``; only
    the ``[p | history]`` tail is copied into this model's
    ``[c (G) | p (P) | history (P*H)]`` normalizer.
    """
    try:
      source = {
        name: source_state_dict[f"obs_normalizer.{name}"]
        for name in ("_mean", "_var", "_std", "count")
      }
    except KeyError as exc:
      raise KeyError(f"HMP checkpoint has no obs-normalizer buffer {exc}.") from exc

    tail_dim = self.proprio_obs_dim * (1 + self.history_steps)
    src_lo = int(source["_mean"].shape[-1]) - tail_dim
    if src_lo < 0:
      raise ValueError(
        f"HMP normalizer ({source['_mean'].shape[-1]}) is smaller than the "
        f"proprio+history tail ({tail_dim})."
      )
    loc_lo = self.task_goal_obs_dim
    normalizer = self.obs_normalizer
    with torch.no_grad():
      for name in ("_mean", "_var", "_std"):
        getattr(normalizer, name)[:, loc_lo : loc_lo + tail_dim] = source[name][
          :, src_lo : src_lo + tail_dim
        ]
      normalizer.count.copy_(source["count"])  # type: ignore[attr-defined]
    print(
      "[HmpLocomotionActorModel] Copied proprio+history obs-normalizer "
      f"from the HMP (count={int(normalizer.count.item())})."  # type: ignore[attr-defined]
    )

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxHmpLocomotionActorModel(self, verbose=verbose)


class _OnnxHmpLocomotionActorModel(nn.Module):
  """ONNX export: deterministic (argmax) code selection -> joint action."""

  is_recurrent: bool = False

  def __init__(self, model: HmpLocomotionActorModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.high_level = copy.deepcopy(model.high_level)
    self.prior = copy.deepcopy(model.prior)
    self.rvq = copy.deepcopy(model.rvq)
    self.action_decoder = copy.deepcopy(model.action_decoder)

    self.input_size = model.obs_dim
    self.task_goal_obs_dim = model.task_goal_obs_dim
    self.current_obs_dim = model.current_obs_dim
    self.history_obs_dim = model.history_obs_dim
    self.num_heads = model.num_active_codebooks
    self.codebook_size = model.codebook_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    g_task = x[:, : self.task_goal_obs_dim]
    proprio = x[:, self.task_goal_obs_dim : self.current_obs_dim]
    history_obs = x[
      :, self.current_obs_dim : self.current_obs_dim + self.history_obs_dim
    ]
    s_rich = torch.cat((proprio, self.history_encoder(history_obs)), dim=-1)
    logits = self.high_level(s_rich, g_task).view(
      -1, self.num_heads, self.codebook_size
    )
    z_hat = self.prior(s_rich) + self.rvq.lookup(logits.argmax(dim=-1))
    return self.action_decoder(s_rich, z_hat)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
