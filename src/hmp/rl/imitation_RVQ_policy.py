"""Phase II model: proprioceptive encoder + reference encoder + RVQ + decoder."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict

from hmp.rl.policy import MotionEncoder, _build_mlp
from hmp.rl.residual_vq import ResidualVQ, RVQCfg


class _PriorNet(nn.Module):
  """Proprioceptive encoder P_theta: z_prop = P_theta(p, eta), Eq. (5)."""

  def __init__(
    self,
    s_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    layer_norm: bool,
  ) -> None:
    super().__init__()
    self.net = _build_mlp(
      input_dim=int(s_dim),
      output_dim=int(latent_dim),
      hidden_dims=hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )

  def forward(self, s: torch.Tensor) -> torch.Tensor:
    return self.net(s)


class _PosteriorNet(nn.Module):
  """Reference encoder E_phi: z_ref = E_phi(p, eta, m), Eq. (4).

  Training-only: discarded after distillation.
  """

  def __init__(
    self,
    s_dim: int,
    goal_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    layer_norm: bool,
  ) -> None:
    super().__init__()
    self.net = _build_mlp(
      input_dim=int(s_dim) + int(goal_dim),
      output_dim=int(latent_dim),
      hidden_dims=hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )

  def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat((s, goal), dim=-1))


class _ActionDecoder(nn.Module):
  """Observation-injected action decoder D_psi(p, eta, z_hat), Eq. (9).

  Every hidden layer receives ``s = [p, eta]`` alongside the latent stream.
  """

  def __init__(
    self,
    s_dim: int,
    latent_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
  ) -> None:
    super().__init__()
    self.activation_fn = resolve_nn_activation(activation)
    layers: list[nn.Module] = []
    in_size = int(latent_dim)
    for h in hidden_dims:
      layers.append(nn.Linear(in_size + int(s_dim), int(h)))
      in_size = int(h)
    self.layers = nn.ModuleList(layers)
    self.mu_head = nn.Linear(in_size, int(action_dim))

  def forward(self, s: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    x = z_hat
    for layer in self.layers:
      x = self.activation_fn(layer(torch.cat((s, x), dim=-1)))
    return self.mu_head(x)


class HmpImitationModel(MLPModel):
  """Distillation student: reference encoder + HMP (Sec. III-B).

  Observation layout (same as the expert): ``[m | p | history]``.

  Forward pass:
    1. eta    = H_alpha(history)                 history embedding, Eq. (3)
    2. s      = [p, eta]
    3. z_prop = P_theta(s)                        Eq. (5)
    4. z_ref  = E_phi(s, m)                       Eq. (4)
    5. y      = z_ref - sg(z_prop)                Eq. (6)
    6. y_hat  = RVQ(y)                            Eq. (7)
    7. z_hat  = sg(z_prop) + y_hat                Eq. (8)
    8. a_hat  = D_psi(s, z_hat)                   Eq. (9)
  """

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
    current_motion_obs_dim: int = 0,
    proprio_obs_dim: int = 0,
    history_steps: int = 10,
    history_latent_dim: int = 128,
    history_conv_channels: tuple[int, ...] | list[int] = (64, 32),
    history_conv_kernel_sizes: tuple[int, ...] | list[int] = (4, 2),
    history_conv_strides: tuple[int, ...] | list[int] = (2, 1),
    latent_dim: int = 128,
    prior_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    posterior_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    layer_norm: bool = True,
    rvq_num_quantizers: int = 8,
    rvq_codebook_size: int = 1024,
    rvq_quantize_dropout: bool = True,
    rvq_decay: float = 0.99,
    rvq_eps: float = 1e-5,
    rvq_commitment_weight: float = 1.0,
    rvq_kmeans_init: bool = False,
    rvq_kmeans_iters: int = 10,
    rvq_rotation_trick: bool = True,
  ) -> None:
    del distribution_cfg  # deterministic model: no action distribution.
    self.current_motion_obs_dim = int(current_motion_obs_dim)
    self.proprio_obs_dim = int(proprio_obs_dim)
    self.history_steps = int(history_steps)
    self.history_latent_dim = int(history_latent_dim)
    self.latent_dim = int(latent_dim)
    self.action_dim = int(output_dim)

    for name in (
      "current_motion_obs_dim",
      "proprio_obs_dim",
      "history_steps",
      "latent_dim",
    ):
      if getattr(self, name) <= 0:
        raise ValueError(f"`{name}` must be positive, got {getattr(self, name)}.")

    self.current_obs_dim = self.current_motion_obs_dim + self.proprio_obs_dim
    self.history_obs_dim = self.proprio_obs_dim * self.history_steps

    # MLPModel always builds `self.mlp`; the action decoder owns the output
    # head, so it is replaced by an Identity right after construction.
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
    self.mlp = nn.Identity()

    expected_obs_dim = self.current_obs_dim + self.history_obs_dim
    if self.obs_dim != expected_obs_dim:
      raise ValueError(
        "HmpImitationModel observation dimension mismatch: "
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
      hidden_dims=prior_hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )
    self.posterior = _PosteriorNet(
      s_dim=s_dim,
      goal_dim=self.current_motion_obs_dim,
      latent_dim=self.latent_dim,
      hidden_dims=posterior_hidden_dims,
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
    self.rvq = ResidualVQ(
      RVQCfg(
        dim=self.latent_dim,
        num_quantizers=int(rvq_num_quantizers),
        codebook_size=int(rvq_codebook_size),
        quantize_dropout=bool(rvq_quantize_dropout),
        decay=float(rvq_decay),
        eps=float(rvq_eps),
        commitment_weight=float(rvq_commitment_weight),
        kmeans_init=bool(rvq_kmeans_init),
        kmeans_iters=int(rvq_kmeans_iters),
        rotation_trick=bool(rvq_rotation_trick),
      )
    )
    print(f"[HmpImitationModel] rotation trick: {bool(rvq_rotation_trick)}")

  def _state_and_goal(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(s, m)`` with ``s = [p, eta]`` from normalized observations."""
    obs_flat = super().get_latent(obs, masks, hidden_state)
    motion_obs = obs_flat[:, : self.current_motion_obs_dim]
    proprio_obs = obs_flat[:, self.current_motion_obs_dim : self.current_obs_dim]
    history_obs = obs_flat[
      :, self.current_obs_dim : self.current_obs_dim + self.history_obs_dim
    ]
    s = torch.cat((proprio_obs, self.history_encoder(history_obs)), dim=-1)
    return s, motion_obs

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> dict[str, torch.Tensor]:
    """Full distillation forward pass; returns all intermediate tensors."""
    del stochastic_output  # kept for MLPModel signature compatibility.
    s, goal = self._state_and_goal(obs, masks, hidden_state)
    zp = self.prior(s)
    z = self.posterior(s, goal)
    zp_sg = zp.detach()
    y_hat, vq_info = self.rvq(z - zp_sg)
    z_hat = zp_sg + y_hat
    return {
      "a_hat": self.action_decoder(s, z_hat),
      "zp": zp,
      "z": z,
      "y_hat": y_hat,
      "z_hat": z_hat,
      "vq_info": vq_info,
    }

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxHmpImitationModel(self, verbose=verbose)


class _OnnxHmpImitationModel(nn.Module):
  """ONNX export of the distilled tracker (reference encoder + HMP).

  Used to evaluate distillation fidelity against the expert (Sec. IV-B): it
  consumes the motion command like the expert does.
  """

  is_recurrent: bool = False

  def __init__(self, model: HmpImitationModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.prior = copy.deepcopy(model.prior)
    self.posterior = copy.deepcopy(model.posterior)
    self.rvq = copy.deepcopy(model.rvq)
    self.action_decoder = copy.deepcopy(model.action_decoder)

    self.input_size = model.obs_dim
    self.current_motion_obs_dim = model.current_motion_obs_dim
    self.current_obs_dim = model.current_obs_dim
    self.history_obs_dim = model.history_obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    motion_obs = x[:, : self.current_motion_obs_dim]
    proprio_obs = x[:, self.current_motion_obs_dim : self.current_obs_dim]
    history_obs = x[
      :, self.current_obs_dim : self.current_obs_dim + self.history_obs_dim
    ]
    s = torch.cat((proprio_obs, self.history_encoder(history_obs)), dim=-1)
    zp = self.prior(s)
    y_hat, _ = self.rvq(self.posterior(s, motion_obs) - zp)
    return self.action_decoder(s, zp + y_hat)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
