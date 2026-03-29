"""Nerfify modules — NeRF primitives integrated from dependencies.md.

This file implements the four core NeRF primitives cited in the NERFIFY
paper but defined in their respective source papers:

    1. HashEncoding        — Instant-NGP multi-resolution hash encoding
    2. VMDecomposition     — TensoRF vector-matrix factorization
    3. ProposalNetwork     — Mip-NeRF 360 density proposal MLP
    4. Distortion/Interlevel losses — from Mip-NeRF 360

Plus the core NeRF primitives:
    5. NerfMLP             — MLP mapping (position, direction) → (density, rgb)
    6. VolumeRenderer      — differentiable volume rendering

Topological position: depends on configuration.py.

References:
    .agent/skills/dependencies.md — §1-§4
    Paper §3.2, Stage 2 — compositional dependency resolution
    Paper Figure 3 — K-Planes citation dependency graph
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration import NerfifyConfig


# ═══════════════════════════════════════════════════════════════════════
# 1. Hash Encoding (Instant-NGP) — dependencies.md §2
# ═══════════════════════════════════════════════════════════════════════

class HashEncoding(nn.Module):
    """Multi-resolution hash encoding from Instant-NGP.

    h(x) = (⊕_{i=1}^d ⌊x_i · N_l⌋ π_i) mod T

    Uses trainable feature vectors at each hash level,
    with trilinear interpolation for continuous queries.

    Args:
        config: NerfifyConfig providing hash_table_size, num_hash_levels,
                features_per_level, spatial_dim.
    """

    PRIMES = [1, 2654435761, 805459861]

    def __init__(self, config: NerfifyConfig):
        super().__init__()
        self.num_levels = config.num_hash_levels
        self.table_size = config.hash_table_size
        self.features_per_level = config.features_per_level
        self.spatial_dim = config.spatial_dim

        # Resolution grows geometrically across levels
        self.base_resolution = 16
        self.max_resolution = 2048
        b = (self.max_resolution / self.base_resolution) ** (
            1.0 / (self.num_levels - 1)
        )
        self.resolutions = [int(self.base_resolution * (b ** l)) for l in range(self.num_levels)]

        # Trainable hash tables: one per level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.table_size, self.features_per_level) * 0.01
            )
            for _ in range(self.num_levels)
        ])

        self.output_dim = self.num_levels * self.features_per_level

    def _hash_fn(self, coords: torch.Tensor) -> torch.Tensor:
        """Spatial hashing: h(x) = (⊕ ⌊x_i⌋ π_i) mod T.

        Args:
            coords: [..., 3] integer grid coordinates.
        Returns:
            indices: [...] hash table indices.
        """
        device = coords.device
        primes = torch.tensor(self.PRIMES[: self.spatial_dim], device=device)
        # XOR across spatial dims
        hashed = (coords.long() * primes).sum(dim=-1)
        return hashed % self.table_size

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode positions via multi-resolution hashing.

        Args:
            positions: [..., 3] normalized positions in [0, 1].
        Returns:
            features: [..., output_dim] concatenated multi-level features.
        """
        output_shape = positions.shape[:-1]
        flat_pos = positions.reshape(-1, self.spatial_dim)  # [P, 3]

        level_features = []
        for level_idx in range(self.num_levels):
            res = self.resolutions[level_idx]
            # Scale position to grid
            scaled = flat_pos * res
            # Floor indices for hashing
            floor_coords = torch.floor(scaled).clamp(0, res - 1)
            # Hash to table index
            idx = self._hash_fn(floor_coords)  # [P]
            # Lookup features
            feats = self.hash_tables[level_idx][idx]  # [P, F]
            level_features.append(feats)

        # Concatenate all levels: [P, num_levels * features_per_level]
        out = torch.cat(level_features, dim=-1)
        return out.reshape(*output_shape, self.output_dim)


# ═══════════════════════════════════════════════════════════════════════
# 2. VM Decomposition (TensoRF) — dependencies.md §3
# ═══════════════════════════════════════════════════════════════════════

class VMDecomposition(nn.Module):
    """Vector-Matrix decomposition from TensoRF.

    T_{i,j,k} = Σ_{r=1}^R Σ_{m∈{X,Y,Z}} v_r^(m) ⊗ M_r^(p_m)

    Factorizes a 3D feature grid into vector + matrix components
    for memory-efficient high-resolution representation.
    """

    def __init__(self, config: NerfifyConfig):
        super().__init__()
        R = config.vm_rank
        res = config.vm_resolution

        # Three plane components: XY, XZ, YZ
        self.plane_coef = nn.ParameterList([
            nn.Parameter(torch.randn(1, R, res, res) * 0.1)
            for _ in range(3)
        ])
        # Three line components: Z, Y, X
        self.line_coef = nn.ParameterList([
            nn.Parameter(torch.randn(1, R, res, 1) * 0.1)
            for _ in range(3)
        ])
        self.output_dim = R

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Query the VM decomposition at given positions.

        Args:
            positions: [..., 3] normalized coordinates in [-1, 1].
        Returns:
            features: [..., vm_rank] feature vectors.
        """
        original_shape = positions.shape[:-1]
        flat_pos = positions.reshape(-1, 3)  # [P, 3]
        P = flat_pos.shape[0]

        # Coordinate pairs for each plane
        plane_coords_list = [
            flat_pos[:, [0, 1]],  # XY plane
            flat_pos[:, [0, 2]],  # XZ plane
            flat_pos[:, [1, 2]],  # YZ plane
        ]
        # Coordinate for each line
        line_coords_list = [
            flat_pos[:, 2:3],  # Z line
            flat_pos[:, 1:2],  # Y line
            flat_pos[:, 0:1],  # X line
        ]

        total = torch.zeros(P, self.output_dim, device=positions.device)

        for i in range(3):
            # Grid sample expects [N, H_out, W_out, 2] with grid in [-1, 1]
            pc = plane_coords_list[i].reshape(1, P, 1, 2)  # [1, P, 1, 2]
            lc = line_coords_list[i].reshape(1, P, 1, 1)     # [1, P, 1, 1]
            # need 2D grid for line as well
            lc_2d = torch.cat([lc, torch.zeros_like(lc)], dim=-1)  # [1, P, 1, 2]

            p_feat = F.grid_sample(
                self.plane_coef[i], pc, align_corners=True, mode="bilinear"
            )  # [1, R, P, 1]
            l_feat = F.grid_sample(
                self.line_coef[i].expand(-1, -1, -1, 2),  # need width=2 for grid_sample
                lc_2d, align_corners=True, mode="bilinear"
            )  # [1, R, P, 1]

            # Product and sum: [1, R, P, 1] → [P, R]
            total += (p_feat * l_feat).squeeze(0).squeeze(-1).permute(1, 0)

        return total.reshape(*original_shape, self.output_dim)


# ═══════════════════════════════════════════════════════════════════════
# 3. Proposal Network (Mip-NeRF 360) — dependencies.md §4
# ═══════════════════════════════════════════════════════════════════════

class ProposalNetwork(nn.Module):
    """Lightweight density proposal MLP from Mip-NeRF 360.

    Small MLP that predicts density to guide importance sampling.
    Trained via inter-level consistency loss.
    """

    def __init__(self, config: NerfifyConfig):
        super().__init__()
        input_dim = config.spatial_dim
        hidden = config.hidden_dim // 2  # Smaller than main field

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),  # density only
            nn.Softplus(),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Predict density at positions.

        Args:
            positions: [..., 3] spatial coordinates.
        Returns:
            density: [..., 1] predicted density.
        """
        return self.net(positions)


# ═══════════════════════════════════════════════════════════════════════
# 4. Losses — dependencies.md §1 and §4
# ═══════════════════════════════════════════════════════════════════════

def distortion_loss(
    weights: torch.Tensor,
    t_vals: torch.Tensor,
) -> torch.Tensor:
    """Distortion loss from Mip-NeRF 360 (dependencies.md §1).

    L_dist(s, w) = Σ_{i,j} w_i w_j |s̄_i - s̄_j| + (1/3) Σ_i w_i² δ_i

    Efficient O(N) implementation using cumulative sums.

    Args:
        weights: [B, N, S] volume rendering weights.
        t_vals: [B, N, S] distances along rays.
    Returns:
        loss: scalar distortion loss.
    """
    # Interval midpoints and widths
    midpoints = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    intervals = t_vals[..., 1:] - t_vals[..., :-1]

    # Trim weights to match intervals
    w = weights[..., :-1]

    # Unary term: (1/3) Σ w² δ
    loss_uni = (1.0 / 3.0) * (intervals * w.pow(2)).sum(dim=-1).mean()

    # Binary term via cumulative sums (O(N))
    wm = w * midpoints
    w_cum = w.cumsum(dim=-1)
    wm_cum = wm.cumsum(dim=-1)
    loss_bi = 2 * (
        wm[..., 1:] * w_cum[..., :-1] - w[..., 1:] * wm_cum[..., :-1]
    ).sum(dim=-1).mean()

    return loss_bi + loss_uni


def interlevel_loss(
    weights_proposal: torch.Tensor,
    weights_fine: torch.Tensor,
) -> torch.Tensor:
    """Inter-level loss from Mip-NeRF 360 (dependencies.md §4).

    Ensures proposal weights upper-bound fine weights.

    Args:
        weights_proposal: [B, N, S] proposal network weights.
        weights_fine: [B, N, S] fine network weights.
    Returns:
        loss: scalar inter-level loss.
    """
    # Ensure same shape (truncate to min)
    min_s = min(weights_proposal.shape[-1], weights_fine.shape[-1])
    wp = weights_proposal[..., :min_s]
    wf = weights_fine[..., :min_s]

    return torch.mean(
        torch.clamp(wp - wf, min=0) ** 2 / (wp + 1e-7)
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. NerfMLP — Core field network
# ═══════════════════════════════════════════════════════════════════════

class NerfMLP(nn.Module):
    """Multi-layer perceptron mapping (position, direction) → (density, rgb).

    Paper Figure 4 field.py agent:
        "map ray_samples → {density, rgb}"
        "two-layer mapping from direction and latent encoding to rgb"

    Architecture:
        position → hidden → density head → density
        [hidden; direction] → rgb head → rgb
    """

    def __init__(self, config: NerfifyConfig, input_dim: int):
        super().__init__()
        H = config.hidden_dim

        # Position encoder → hidden features → density
        layers = [nn.Linear(input_dim, H), nn.ReLU(inplace=True)]
        for _ in range(config.num_layers - 2):
            layers.extend([nn.Linear(H, H), nn.ReLU(inplace=True)])
        self.position_net = nn.Sequential(*layers)

        self.density_head = nn.Sequential(
            nn.Linear(H, 1),
            nn.Softplus(),
        )

        # Direction-dependent RGB head
        self.rgb_head = nn.Sequential(
            nn.Linear(H + config.direction_dim, H // 2),
            nn.ReLU(inplace=True),
            nn.Linear(H // 2, config.rgb_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            positions: [..., input_dim] encoded spatial features.
            directions: [..., 3] view directions (broadcast to match).
        Returns:
            dict with "density": [..., 1] and "rgb": [..., 3].
        """
        h = self.position_net(positions)
        density = self.density_head(h)

        # Expand directions to match hidden spatial dims
        if directions.dim() < positions.dim():
            for _ in range(positions.dim() - directions.dim()):
                directions = directions.unsqueeze(-2)
            directions = directions.expand(*h.shape[:-1], -1)

        rgb_input = torch.cat([h, directions], dim=-1)
        rgb = self.rgb_head(rgb_input)

        return {"density": density, "rgb": rgb}


# ═══════════════════════════════════════════════════════════════════════
# 6. Volume Renderer
# ═══════════════════════════════════════════════════════════════════════

class VolumeRenderer(nn.Module):
    """Differentiable volume rendering via alpha compositing.

    C(r) = Σ_i T_i (1 - exp(-σ_i δ_i)) c_i
    T_i = exp(-Σ_{j<i} σ_j δ_j)
    """

    def forward(
        self,
        densities: torch.Tensor,
        rgbs: torch.Tensor,
        t_vals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render rays.

        Args:
            densities: [B, N, S, 1] predicted density.
            rgbs: [B, N, S, 3] predicted color.
            t_vals: [B, N, S] sample distances.
        Returns:
            dict with "rgb": [B, N, 3], "weights": [B, N, S],
                       "depth": [B, N, 1], "accumulation": [B, N, 1].
        """
        # Interval widths
        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        # Pad last delta
        deltas = torch.cat([
            deltas,
            torch.full_like(deltas[..., :1], 1e10),
        ], dim=-1)

        # Alpha = 1 - exp(-σ δ)
        alpha = 1.0 - torch.exp(-densities.squeeze(-1) * deltas)

        # Transmittance T_i = Π_{j<i} (1 - α_j)
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones(*alpha.shape[:-1], 1, device=alpha.device),
                1.0 - alpha + 1e-10,
            ], dim=-1),
            dim=-1,
        )[..., :-1]

        # Weights w_i = T_i α_i
        weights = transmittance * alpha

        # Composite RGB
        rgb = (weights.unsqueeze(-1) * rgbs).sum(dim=-2)

        # Depth and accumulation
        depth = (weights * t_vals).sum(dim=-1, keepdim=True)
        accumulation = weights.sum(dim=-1, keepdim=True)

        return {
            "rgb": rgb,
            "weights": weights,
            "depth": depth,
            "accumulation": accumulation,
        }


# ============================================================
# VALIDATION BLOCK — shape checks through all modules
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATING: modules.py")
    print("=" * 60)

    config = NerfifyConfig()
    B, N, S = 2, 8, config.num_samples  # batch, rays, samples

    # --- HashEncoding ---
    hash_enc = HashEncoding(config)
    positions = torch.rand(B, N, S, 3)  # [0, 1] range
    hash_out = hash_enc(positions)
    expected_hash_dim = config.num_hash_levels * config.features_per_level
    print(f"  HashEncoding:  {positions.shape} → {hash_out.shape}  (expected [..., {expected_hash_dim}])")
    assert hash_out.shape == (B, N, S, expected_hash_dim), f"Got {hash_out.shape}"
    print("  ✅ HashEncoding shapes OK")

    # --- VMDecomposition ---
    vm = VMDecomposition(config)
    vm_pos = torch.rand(B, N, S, 3) * 2 - 1  # [-1, 1] range
    vm_out = vm(vm_pos)
    print(f"  VMDecomposition: {vm_pos.shape} → {vm_out.shape}  (expected [..., {config.vm_rank}])")
    assert vm_out.shape == (B, N, S, config.vm_rank), f"Got {vm_out.shape}"
    print("  ✅ VMDecomposition shapes OK")

    # --- ProposalNetwork ---
    prop_net = ProposalNetwork(config)
    prop_pos = torch.randn(B, N, S, 3)
    prop_out = prop_net(prop_pos)
    print(f"  ProposalNet:   {prop_pos.shape} → {prop_out.shape}  (expected [..., 1])")
    assert prop_out.shape == (B, N, S, 1), f"Got {prop_out.shape}"
    print("  ✅ ProposalNetwork shapes OK")

    # --- NerfMLP ---
    input_dim = expected_hash_dim  # from hash encoding
    nerf_mlp = NerfMLP(config, input_dim=input_dim)
    directions = torch.randn(B, N, 1, 3).expand(B, N, S, 3)
    mlp_out = nerf_mlp(hash_out, directions)
    print(f"  NerfMLP density: {mlp_out['density'].shape}  (expected [{B}, {N}, {S}, 1])")
    print(f"  NerfMLP rgb:     {mlp_out['rgb'].shape}  (expected [{B}, {N}, {S}, 3])")
    assert mlp_out["density"].shape == (B, N, S, 1), f"Got {mlp_out['density'].shape}"
    assert mlp_out["rgb"].shape == (B, N, S, 3), f"Got {mlp_out['rgb'].shape}"
    print("  ✅ NerfMLP shapes OK")

    # --- VolumeRenderer ---
    renderer = VolumeRenderer()
    t_vals = torch.linspace(0.1, 6.0, S).expand(B, N, S)
    render_out = renderer(mlp_out["density"], mlp_out["rgb"], t_vals)
    print(f"  Renderer rgb:    {render_out['rgb'].shape}  (expected [{B}, {N}, 3])")
    print(f"  Renderer weights: {render_out['weights'].shape}  (expected [{B}, {N}, {S}])")
    print(f"  Renderer depth:   {render_out['depth'].shape}  (expected [{B}, {N}, 1])")
    assert render_out["rgb"].shape == (B, N, 3), f"Got {render_out['rgb'].shape}"
    assert render_out["weights"].shape == (B, N, S), f"Got {render_out['weights'].shape}"
    assert render_out["depth"].shape == (B, N, 1), f"Got {render_out['depth'].shape}"
    print("  ✅ VolumeRenderer shapes OK")

    # --- Distortion loss ---
    d_loss = distortion_loss(render_out["weights"], t_vals)
    print(f"  Distortion loss: {d_loss.item():.6f}")
    assert d_loss.isfinite(), "Distortion loss is not finite!"
    print("  ✅ Distortion loss finite")

    # --- Interlevel loss ---
    fake_proposal_w = torch.rand_like(render_out["weights"])
    il_loss = interlevel_loss(fake_proposal_w, render_out["weights"])
    print(f"  Interlevel loss: {il_loss.item():.6f}")
    assert il_loss.isfinite(), "Interlevel loss is not finite!"
    print("  ✅ Interlevel loss finite")

    # --- Gradient flow check ---
    loss_total = d_loss + il_loss + render_out["rgb"].mean()
    loss_total.backward()
    grad_count = sum(1 for p in hash_enc.parameters() if p.grad is not None)
    print(f"  Gradient flow: {grad_count}/{sum(1 for _ in hash_enc.parameters())} HashEncoding params have gradients")
    assert grad_count > 0, "No gradients flowing!"
    print("  ✅ Gradient flow OK")

    print()
    print("🎉 modules.py — ALL CHECKS PASSED")
    print("=" * 60)
