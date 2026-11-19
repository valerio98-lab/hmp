from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


@dataclass
class RVQCfg:
  dim: int
  num_quantizers: int = 8
  codebook_size: int = 1024
  # Quantizer dropout (Sec. III-B.2): during training evaluate Eq. (7) over a
  # random prefix of M' ~ U{1..M} stages.
  quantize_dropout: bool = True
  # VectorQuantize kwargs. Codebooks are EMA-updated (decay) and trained with
  # the commitment loss of Eq. (14) (beta = commitment_weight).
  decay: float = 0.99
  eps: float = 1e-5
  commitment_weight: float = 1.0
  kmeans_init: bool = False
  kmeans_iters: int = 10
  # Rotation trick (Sec. III-B.4) instead of the straight-through estimator.
  rotation_trick: bool = True


class ResidualVQ(nn.Module):
  """M-stage residual vector quantizer for latents of shape (B, D).

  Each stage quantizes the residual left by the previous stages; the selected
  code vectors are summed (Eq. 7).

  ``forward`` returns ``(y_hat, info)`` where ``info`` holds the per-stage
  ``indices`` (B, M'), the commitment loss ``loss_vq`` and codebook-usage
  statistics.
  """

  def __init__(self, cfg: RVQCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.dim = int(cfg.dim)
    self.num_quantizers = int(cfg.num_quantizers)
    self.codebook_size = int(cfg.codebook_size)
    self.quantize_dropout = bool(cfg.quantize_dropout) and self.num_quantizers > 1
    self.layers = nn.ModuleList(
      VectorQuantize(
        dim=self.dim,
        codebook_size=self.codebook_size,
        decay=float(cfg.decay),
        eps=float(cfg.eps),
        commitment_weight=float(cfg.commitment_weight),
        kmeans_init=bool(cfg.kmeans_init),
        kmeans_iters=int(cfg.kmeans_iters),
        rotation_trick=bool(cfg.rotation_trick),
      )
      for _ in range(self.num_quantizers)
    )

  def _sample_num_active(self) -> int:
    """Number of active stages M': all at eval, U{1..M} with dropout."""
    if not self.training or not self.quantize_dropout:
      return self.num_quantizers
    return int(torch.randint(1, self.num_quantizers + 1, (1,)).item())

  def lookup(self, indices: torch.Tensor) -> torch.Tensor:
    """Sum the code vectors selected by ``indices`` (B, M_act), Eq. (7).

    Uses the first ``indices.shape[-1]`` stages. Pure table lookup: no
    gradient to the codebooks and no EMA update.
    """
    num_active = int(indices.shape[-1])
    if indices.dim() != 2 or not 1 <= num_active <= self.num_quantizers:
      raise ValueError(
        f"lookup expected indices of shape (B, 1..{self.num_quantizers}), "
        f"got {tuple(indices.shape)}."
      )
    y_hat = torch.zeros(
      indices.shape[0], self.dim, device=indices.device, dtype=torch.float32
    )
    for qi in range(num_active):
      codebook = self.layers[qi]._codebook.embed[0]  # (K, D)
      y_hat = y_hat + codebook[indices[:, qi]]
    return y_hat

  def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if y.dim() != 2 or y.shape[-1] != self.dim:
      raise ValueError(f"Expected (B, {self.dim}), got {tuple(y.shape)}.")

    residual = y
    y_hat = torch.zeros_like(y)
    num_active = self._sample_num_active()
    all_indices: list[torch.Tensor] = []
    all_losses: list[torch.Tensor] = []

    for layer in self.layers[:num_active]:
      quantized, indices, loss = layer(residual)
      # Detach so each stage only receives the residual left by the
      # previous stages (no gradient through earlier code choices).
      residual = residual - quantized.detach()
      y_hat = y_hat + quantized
      all_indices.append(indices)
      all_losses.append(loss.reshape(()))

    losses_per_layer = torch.stack(all_losses)
    perplexities, usages = [], []
    for layer_indices in all_indices:
      counts = torch.bincount(
        layer_indices.flatten(), minlength=self.codebook_size
      ).float()
      probs = counts / counts.sum().clamp_min(1.0)
      nz = probs > 0
      perplexities.append((-(probs[nz] * probs[nz].log()).sum()).exp())
      usages.append(nz.sum().to(y.dtype))
    usages_per_layer = torch.stack(usages)

    info = {
      "indices": torch.stack(all_indices, dim=-1),  # (B, M')
      "loss_vq": losses_per_layer.mean(),  # Eq. (14): 1/M' sum over stages
      "losses_per_layer": losses_per_layer,
      "num_active": torch.tensor(num_active, device=y.device),
      "perplexities_per_layer": torch.stack(perplexities),
      "usages_per_layer": usages_per_layer,
      "usage_ratios_per_layer": usages_per_layer / float(self.codebook_size),
    }
    return y_hat, info
