"""NerfifyDataset — Synthetic ray dataset for training and validation.

Generates ray bundles (origins, directions, near/far bounds) compatible
with the Nerfify model.  Maps NeRF-style input to HF-compatible
(input_ids, attention_mask, labels) tensors for the forward() contract.

Topological position: depends on configuration.py only.

References:
    Paper Figure 4 — DataManager / batch generation
    Paper §4.2 — Blender and DTU datasets
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ── dependency (topo level 0) ───────────────────────────────────────────
from configuration import NerfifyConfig


class NerfifyRayBundle:
    """A batch of rays for NeRF rendering.

    Attributes:
        origins: [B, N, 3] ray origin points.
        directions: [B, N, 3] unit direction vectors.
        nears: [B, N, 1] near plane distances.
        fars: [B, N, 1] far plane distances.
        pixel_rgb: [B, N, 3] ground-truth RGB for training (optional).
    """

    def __init__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        nears: torch.Tensor,
        fars: torch.Tensor,
        pixel_rgb: Optional[torch.Tensor] = None,
    ):
        self.origins = origins
        self.directions = directions
        self.nears = nears
        self.fars = fars
        self.pixel_rgb = pixel_rgb

    def sample_along_rays(self, num_samples: int) -> torch.Tensor:
        """Uniform stratified sampling along rays.

        Returns:
            t_vals: [B, N, num_samples] distances along each ray.
        """
        B, N, _ = self.origins.shape
        device = self.origins.device

        # Stratified sampling between near and far
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        t_vals = t_vals.unsqueeze(0).unsqueeze(0).expand(B, N, num_samples)

        # Map [0,1] to [near, far]
        t_vals = self.nears * (1.0 - t_vals) + self.fars * t_vals

        # Add uniform jitter for stochastic sampling during training
        if self.origins.requires_grad or True:  # always jitter for now
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        return t_vals

    def get_points(self, t_vals: torch.Tensor) -> torch.Tensor:
        """Compute 3D points along rays.

        Args:
            t_vals: [B, N, S] distances along rays.

        Returns:
            points: [B, N, S, 3] 3D sample points.
        """
        # origins: [B, N, 3] -> [B, N, 1, 3]
        # directions: [B, N, 3] -> [B, N, 1, 3]
        # t_vals: [B, N, S] -> [B, N, S, 1]
        return (
            self.origins.unsqueeze(2)
            + self.directions.unsqueeze(2) * t_vals.unsqueeze(3)
        )


class NerfifyDataset(Dataset):
    """Synthetic ray dataset that generates random camera rays.

    For actual training you'd load real camera poses and images;
    this synthetic version is for smoke-testing the pipeline shape
    contracts end-to-end.

    Args:
        config: NerfifyConfig providing num_samples and spatial_dim.
        num_rays: Total number of rays in the dataset.
        image_height: Simulated image height for pixel coordinates.
        image_width: Simulated image width for pixel coordinates.
    """

    def __init__(
        self,
        config: NerfifyConfig,
        num_rays: int = 1024,
        image_height: int = 64,
        image_width: int = 64,
    ):
        super().__init__()
        self.config = config
        self.num_rays = num_rays
        self.image_height = image_height
        self.image_width = image_width

        # Pre-generate random camera poses (rotation + translation)
        self.num_cameras = max(1, num_rays // (image_height * image_width))
        self.camera_positions = self._random_camera_positions(self.num_cameras)

    def _random_camera_positions(self, n: int) -> torch.Tensor:
        """Generate n random camera positions on a sphere of radius 4."""
        theta = torch.rand(n) * 2 * math.pi
        phi = torch.acos(1 - 2 * torch.rand(n))
        radius = 4.0
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        return torch.stack([x, y, z], dim=-1)  # [n, 3]

    def __len__(self) -> int:
        return self.num_rays

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single ray with synthetic GT RGB.

        Returns dict with keys matching the HF convention:
            input_ids: [spatial_dim + direction_dim] flattened ray params
            attention_mask: [spatial_dim + direction_dim] all ones
            labels: [rgb_dim] ground-truth pixel colour
        """
        cam_idx = idx % self.num_cameras
        origin = self.camera_positions[cam_idx]

        # Random direction (normalized)
        direction = torch.randn(self.config.direction_dim)
        direction = direction / direction.norm()

        # Flatten origin + direction as input_ids (continuous, but fits contract)
        input_ids = torch.cat([origin, direction], dim=0)  # [6]
        attention_mask = torch.ones_like(input_ids)         # [6]

        # Synthetic GT: random RGB
        labels = torch.rand(self.config.rgb_dim)  # [3]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def collate_as_ray_bundle(
        self,
        batch_indices: torch.Tensor,
    ) -> NerfifyRayBundle:
        """Collate a batch of rays into a NerfifyRayBundle.

        Args:
            batch_indices: [B] tensor of dataset indices.

        Returns:
            NerfifyRayBundle with shapes [1, B, ...].
        """
        items = [self[idx.item()] for idx in batch_indices]
        origins = torch.stack([item["input_ids"][:3] for item in items])  # [B, 3]
        directions = torch.stack([item["input_ids"][3:] for item in items])  # [B, 3]
        pixel_rgb = torch.stack([item["labels"] for item in items])  # [B, 3]

        B = origins.shape[0]
        nears = torch.full((B, 1), 0.1)
        fars = torch.full((B, 1), 6.0)

        # Add batch dimension: [1, B, ...]
        return NerfifyRayBundle(
            origins=origins.unsqueeze(0),
            directions=directions.unsqueeze(0),
            nears=nears.unsqueeze(0),
            fars=fars.unsqueeze(0),
            pixel_rgb=pixel_rgb.unsqueeze(0),
        )


# ============================================================
# VALIDATION BLOCK — dummy tensor shape checks
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATING: dataset.py")
    print("=" * 60)

    config = NerfifyConfig()
    dataset = NerfifyDataset(config, num_rays=256)
    print(f"  Dataset length: {len(dataset)}")

    # Test 1: Single item shape
    item = dataset[0]
    print(f"  input_ids shape:     {item['input_ids'].shape}  (expected [6])")
    print(f"  attention_mask shape: {item['attention_mask'].shape}  (expected [6])")
    print(f"  labels shape:        {item['labels'].shape}  (expected [3])")
    assert item["input_ids"].shape == (6,), f"Got {item['input_ids'].shape}"
    assert item["attention_mask"].shape == (6,), f"Got {item['attention_mask'].shape}"
    assert item["labels"].shape == (3,), f"Got {item['labels'].shape}"
    print("  ✅ Single item shapes OK")

    # Test 2: Ray bundle collation
    batch_idx = torch.arange(8)
    bundle = dataset.collate_as_ray_bundle(batch_idx)
    print(f"  origins shape:    {bundle.origins.shape}  (expected [1, 8, 3])")
    print(f"  directions shape: {bundle.directions.shape}  (expected [1, 8, 3])")
    print(f"  nears shape:      {bundle.nears.shape}  (expected [1, 8, 1])")
    print(f"  fars shape:       {bundle.fars.shape}  (expected [1, 8, 1])")
    print(f"  pixel_rgb shape:  {bundle.pixel_rgb.shape}  (expected [1, 8, 3])")
    assert bundle.origins.shape == (1, 8, 3)
    assert bundle.directions.shape == (1, 8, 3)
    assert bundle.nears.shape == (1, 8, 1)
    assert bundle.fars.shape == (1, 8, 1)
    assert bundle.pixel_rgb.shape == (1, 8, 3)
    print("  ✅ Ray bundle shapes OK")

    # Test 3: Stratified sampling
    t_vals = bundle.sample_along_rays(config.num_samples)
    print(f"  t_vals shape:     {t_vals.shape}  (expected [1, 8, {config.num_samples}])")
    assert t_vals.shape == (1, 8, config.num_samples)
    print("  ✅ Stratified sampling shapes OK")

    # Test 4: 3D point computation
    points = bundle.get_points(t_vals)
    print(f"  points shape:     {points.shape}  (expected [1, 8, {config.num_samples}, 3])")
    assert points.shape == (1, 8, config.num_samples, 3)
    print("  ✅ 3D points shapes OK")

    # Test 5: Matrix multiply compatibility — points through a linear layer
    import torch.nn as nn
    linear = nn.Linear(3, config.hidden_dim)
    out = linear(points)
    print(f"  linear(points) shape: {out.shape}  (expected [1, 8, {config.num_samples}, {config.hidden_dim}])")
    assert out.shape == (1, 8, config.num_samples, config.hidden_dim)
    print("  ✅ MatMul compatibility OK")

    print()
    print("🎉 dataset.py — ALL CHECKS PASSED")
    print("=" * 60)
