from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class CARSState:
    """Per-layer recurrent relational state: [batch, heads, value_dim, key_dim]."""

    matrix: Tensor

    def detach(self) -> "CARSState":
        return CARSState(self.matrix.detach())


@dataclass
class CARSDiagnostics:
    state_change: Tensor       # [B, T, H]
    innovation_norm: Tensor    # [B, T, H]
    read_norm: Tensor          # [B, T, H]
    transport_strength: Tensor  # [B, T, H], Cayley generator magnitude


def _inv_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("inverse softplus input must be positive")
    return math.log(math.expm1(x))


class CARSStateMixer(nn.Module):
    """Corrective Affine Relational State (CARS) mixer.

    One input-conditioned operator performs:
      1) differentiable low-rank orthogonal Cayley transport of relational state,
      2) strict read-before-write retrieval from transported history,
      3) rank-r ridge minimum-change correction from current evidence.

    The recurrent reference arithmetic is deliberately FP32 for numerical
    auditability. There are no token-level routing, write, erase, planner,
    verifier, or refinement controllers.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.h = config.n_heads
        self.dk = config.cars_key_dim
        self.dv = config.cars_value_dim
        self.rank = config.cars_evidence_rank

        self.per_head = (
            self.dk                           # q
            + self.dk * self.rank             # K evidence lanes
            + self.dv * self.rank             # V evidence lanes
            + self.dk                         # transport u
            + self.dk                         # transport w
            + 1                               # transport generator strength
        )
        self.in_proj = nn.Linear(config.d_model, self.h * self.per_head, bias=False)
        self.out_proj = nn.Linear(self.h * self.dv, config.d_model, bias=False)

        # Static learned state timescales. We deliberately initialize heads at
        # different horizons so the model does not depend on tiny gradients to
        # discover temporal diversity from an identical starting point.
        half_lives = torch.logspace(
            math.log10(config.cars_half_life_min),
            math.log10(config.cars_half_life_max),
            steps=self.h,
            dtype=torch.float32,
        )
        rates = math.log(2.0) / half_lives
        self.raw_decay_rate = nn.Parameter(torch.tensor([_inv_softplus(float(r)) for r in rates]))

        ridge_extra = config.cars_ridge_init - config.cars_min_ridge
        self.raw_ridge = nn.Parameter(
            torch.full((self.h,), _inv_softplus(ridge_extra), dtype=torch.float32)
        )

        self.raw_read_scale = nn.Parameter(
            torch.full((self.h,), _inv_softplus(config.cars_read_scale_init), dtype=torch.float32)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def apply_architecture_initialization(self) -> None:
        """Apply architecture-specific initialization after global model init.

        Only the transport-strength rows are made near-zero. Geometry vectors
        stay normally initialized; the operator therefore begins close to
        identity transport and must learn to activate non-trivial movement.
        """
        with torch.no_grad():
            rows = torch.arange(self.h, device=self.in_proj.weight.device) * self.per_head + (self.per_head - 1)
            init = torch.empty(
                self.h, self.in_proj.weight.shape[1],
                device=self.in_proj.weight.device, dtype=self.in_proj.weight.dtype,
            ).normal_(mean=0.0, std=self.config.cars_transport_init_std)
            self.in_proj.weight.index_copy_(0, rows, init)
            self.raw_read_scale.fill_(_inv_softplus(self.config.cars_read_scale_init))

    @property
    def state_decay(self) -> Tensor:
        return torch.exp(-F.softplus(self.raw_decay_rate))

    @property
    def state_half_life(self) -> Tensor:
        rate = F.softplus(self.raw_decay_rate).clamp_min(1e-12)
        return math.log(2.0) / rate

    @property
    def ridge(self) -> Tensor:
        return F.softplus(self.raw_ridge) + self.config.cars_min_ridge

    @property
    def read_scale(self) -> Tensor:
        return F.softplus(self.raw_read_scale)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> CARSState:
        # The reference recurrence is FP32 by design. `dtype` is accepted for a
        # stable public signature and for future kernel implementations.
        _ = dtype
        return CARSState(
            torch.zeros(
                batch_size,
                self.h,
                self.dv,
                self.dk,
                device=device,
                dtype=torch.float32,
            )
        )

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self._split_projected(self.in_proj(x))

    def _split_projected(self, projected: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        prefix = projected.shape[:-1]
        p = projected.reshape(*prefix, self.h, self.per_head)
        cursor = 0

        q = p[..., cursor : cursor + self.dk]
        cursor += self.dk

        k = p[..., cursor : cursor + self.dk * self.rank]
        k = k.reshape(*prefix, self.h, self.dk, self.rank)
        cursor += self.dk * self.rank

        v = p[..., cursor : cursor + self.dv * self.rank]
        v = v.reshape(*prefix, self.h, self.dv, self.rank)
        cursor += self.dv * self.rank

        u = p[..., cursor : cursor + self.dk]
        cursor += self.dk
        w = p[..., cursor : cursor + self.dk]
        cursor += self.dk
        strength_raw = p[..., cursor : cursor + 1]

        # Address geometry is scale-free. Reference recurrent arithmetic and
        # the tiny ridge solve are intentionally FP32.
        q = F.normalize(q.float(), dim=-1, eps=1e-6)
        k = F.normalize(k.float(), dim=-2, eps=1e-6)
        u = F.normalize(u.float(), dim=-1, eps=1e-6)
        w = F.normalize(w.float(), dim=-1, eps=1e-6)
        return q, k, v.float(), u, w, strength_raw.float()

    def _transport(
        self,
        state: Tensor,
        u: Tensor,
        w: Tensor,
        strength_raw: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply an orthogonal Cayley transform without a hard fallback.

        A = eta (u w^T - w u^T) is skew-symmetric. The Cayley map

            T = (I - A/2)^-1 (I + A/2)

        is orthogonal in exact arithmetic. We apply S @ T using a 2x2 Woodbury
        solve, so no dense dk x dk transport matrix is materialized. If u and w
        collapse to the same direction, A smoothly becomes zero while gradients
        with respect to their relative direction remain defined.
        """
        if not self.config.cars_transport_enabled or self.config.cars_transport_max_strength == 0.0:
            strength = torch.zeros_like(strength_raw[..., 0])
            return state, strength

        eta = torch.tanh(strength_raw[..., 0]) * self.config.cars_transport_max_strength

        # A = U V^T with U = eta [u, -w], V = [w, u].
        u2 = torch.stack((u, -w), dim=-1) * eta[..., None, None]  # [B,H,dk,2]
        v2 = torch.stack((w, u), dim=-1)                          # [B,H,dk,2]

        # G = I - 1/2 V^T U. For skew A, G is nonsingular for real eta.
        vtu = torch.einsum("bhkq,bhkr->bhqr", v2, u2)
        eye = torch.eye(2, device=state.device, dtype=torch.float32)
        g = eye[None, None] - 0.5 * vtu

        # S @ T = S + (S U) G^-1 V^T.
        su = torch.einsum("bhvk,bhkq->bhvq", state, u2)
        coeff_t = torch.linalg.solve(g.transpose(-1, -2), su.transpose(-1, -2))
        coeff = coeff_t.transpose(-1, -2)
        delta = torch.einsum("bhvq,bhkq->bhvk", coeff, v2)
        return state + delta, eta.abs()

    def _correct(self, state: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        pred = torch.einsum("bhvk,bhkr->bhvr", state, k)
        innovation = v - pred
        if self.config.write_mode == "none":
            return state, innovation
        if self.config.write_mode == "additive":
            # Shared live-write control: no prediction-error correction or solve.
            return state + torch.matmul(v, k.transpose(-1, -2)) / self.rank, innovation

        gram = torch.einsum("bhkr,bhks->bhrs", k, k)
        eye = torch.eye(self.rank, device=state.device, dtype=torch.float32)
        gram = gram + self.ridge[None, :, None, None] * eye[None, None, :, :]

        # Solve G Z = E^T, then delta = Z^T K^T. Cholesky is stable and the
        # system is tiny (rank usually 1-4). No explicit inverse is formed.
        chol = torch.linalg.cholesky(gram)
        z = torch.cholesky_solve(innovation.transpose(-1, -2), chol)
        correction = torch.matmul(z.transpose(-1, -2), k.transpose(-1, -2))
        return state + correction, innovation

    def step(
        self,
        x_t: Tensor,
        state: CARSState,
        *,
        active: Tensor | None = None,
    ) -> tuple[Tensor, CARSState, dict[str, Tensor]]:
        """One causal recurrent step."""
        # Keep projections under the caller's autocast; disable it for recurrence.
        projected = self.in_proj(x_t)
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            read, next_state, diagnostics = self._recurrent_step(
                self._split_projected(projected), state, active=active, return_diagnostics=True
            )
            read = self._process_read(read)
        y = self.out_proj(read.reshape(x_t.shape[0], self.h * self.dv).to(x_t.dtype))
        return y, next_state, diagnostics

    def _process_read(self, read: Tensor) -> Tensor:
        if self.config.read_mode == "rmsnorm":
            read = read * torch.rsqrt(read.square().mean(dim=-1, keepdim=True) + 1e-6)
        elif self.config.read_mode == "rmsclip":
            read = read * torch.rsqrt(1.0 + read.square().mean(dim=-1, keepdim=True) / self.config.cars_read_clip**2)
        return read * self.read_scale[:, None]

    def _recurrent_step(
        self,
        quantities: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        state: CARSState,
        *,
        active: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, CARSState, dict[str, Tensor]]:
        q, k, v, u, w, strength_raw = quantities
        prev = state.matrix.float()
        transported, strength = self._transport(prev, u, w, strength_raw)
        transported = transported * self.state_decay[None, :, None, None]

        # Strict causal semantics: read transformed history before current
        # evidence is written into state.
        read_state = transported
        if self.config.cars_read_before_write:
            read = torch.einsum("bhvk,bhk->bhv", read_state, q)
        write_base = prev * self.state_decay[None, :, None, None] if self.config.cars_transport_read_only else transported
        updated, innovation = self._correct(write_base, k, v)
        if not self.config.cars_read_before_write:
            post = self._transport(updated, u, w, strength_raw)[0] if self.config.cars_transport_read_only else updated
            read = torch.einsum("bhvk,bhk->bhv", post, q)

        if active is not None:
            m_state = active[:, None, None, None]
            updated = torch.where(m_state, updated, prev)
            read = torch.where(active[:, None, None], read, torch.zeros_like(read))

        if not return_diagnostics:
            return read, CARSState(updated), {}
        prev_norm = prev.flatten(2).norm(dim=-1)
        next_norm = updated.flatten(2).norm(dim=-1)
        denom = torch.maximum(prev_norm, next_norm).clamp_min(1e-6)
        change = (updated - prev).flatten(2).norm(dim=-1) / denom
        diagnostics = {
            "state_change": change,
            "innovation_norm": innovation.flatten(2).norm(dim=-1),
            "read_norm": read.norm(dim=-1),
            "transport_strength": strength,
        }
        return read, CARSState(updated), diagnostics

    def forward(
        self,
        x: Tensor,
        *,
        state: CARSState | None = None,
        attention_mask: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, CARSState, CARSDiagnostics | None]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, d_model]")
        b, t, _ = x.shape
        if t == 0:
            raise ValueError("sequence must contain at least one token")
        if state is None:
            state = self.initial_state(b, device=x.device, dtype=x.dtype)
        if attention_mask is not None and attention_mask.shape != (b, t):
            raise ValueError("attention_mask must have shape [batch, time]")
        if state.matrix.shape != (b, self.h, self.dv, self.dk) or state.matrix.device != x.device:
            raise ValueError("state shape/device does not match the input and CARS geometry")

        outputs: list[Tensor] = []
        traces: dict[str, list[Tensor]] = {
            "state_change": [],
            "innovation_norm": [],
            "read_norm": [],
            "transport_strength": [],
        }
        # Two sequence-sized GEMMs replace 2*T small projection GEMMs. Only
        # state evolution is sequential; all sensitive math stays truly FP32.
        projected = self.in_proj(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            quantities = self._split_projected(projected)
            for i in range(t):
                active = None if attention_mask is None else attention_mask[:, i].bool()
                read, state, diag = self._recurrent_step(
                    tuple(value[:, i] for value in quantities), state,
                    active=active, return_diagnostics=return_diagnostics,
                )
                outputs.append(read)
                if return_diagnostics:
                    for key in traces:
                        traces[key].append(diag[key])
            reads = self._process_read(torch.stack(outputs, dim=1))
        y_all = self.out_proj(reads.reshape(b, t, self.h * self.dv).to(x.dtype))
        if not return_diagnostics:
            return y_all, state, None
        diagnostics = CARSDiagnostics(
            state_change=torch.stack(traces["state_change"], dim=1),
            innovation_norm=torch.stack(traces["innovation_norm"], dim=1),
            read_norm=torch.stack(traces["read_norm"], dim=1),
            transport_strength=torch.stack(traces["transport_strength"], dim=1),
        )
        return y_all, state, diagnostics
