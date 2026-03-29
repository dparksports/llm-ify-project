---
name: NeRF Mathematical Dependencies
description: Standard PyTorch implementations and mathematical formulations for NeRF primitives cited in NERFIFY but not explicitly defined.
---

# NeRF Mathematical Dependencies

This document provides the standard PyTorch implementations and mathematical formulations for the architectural primitives, normalization layers, and loss functions explicitly cited in the NERFIFY paper (arXiv:2603.00805v1) but without provided mathematical formulations.

## 1. Distortion Loss (from Mip-NeRF 360)

**Mathematical Formulation:**
The distortion loss encourages the weights $w$ along a ray to be compact and representative of a single surface, penalizing "floaters" or overlapping intervals.
$$ \mathcal{L}_{dist}(\mathbf{s}, \mathbf{w}) = \sum_{i,j} w_i w_j | \bar{s}_i - \bar{s}_j | + \frac{1}{3} \sum_i w_i^2 \delta_i $$
Where:
- $w_i, w_j$ are the volume rendering weights for intervals $i$ and $j$.
- $\bar{s}_i, \bar{s}_j$ are the midpoints of the intervals: $\bar{s}_i = (s_i + s_{i+1})/2$.
- $\delta_i$ is the width of the interval: $\delta_i = s_{i+1} - s_i$.

**PyTorch Implementation (Efficient O(N)):**
```python
import torch

def distortion_loss(weights, midpoints, intervals):
    """
    Efficient O(N) implementation of distortion loss.
    weights: [B, N] volume rendering weights
    midpoints: [B, N] interval midpoints
    intervals: [B, N] interval widths (deltas)
    """
    loss_uni = (1/3) * (intervals * weights.pow(2)).sum(dim=-1).mean()
    wm = (weights * midpoints)
    w_cum = weights.cumsum(dim=-1)
    wm_cum = wm.cumsum(dim=-1)
    
    loss_bi_0 = wm[..., 1:] * w_cum[..., :-1]
    loss_bi_1 = weights[..., 1:] * wm_cum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni
```

## 2. Hash Encoding (from Instant-NGP)

**Mathematical Formulation:**
Uses a multiresolution hierarchy of hash tables to map spatial coordinates to trainable feature vectors. For a level $l$ with resolution $N_l$:
$$ h(\mathbf{x}) = \left( \bigoplus_{i=1}^d \lfloor x_i \cdot N_l \rfloor \pi_i \right) \bmod T $$
Where:
- $\mathbf{x} \in [0, 1]^d$ is the input coordinate.
- $\pi_1=1, \pi_2=2654435761, \pi_3=805459861$ are large prime numbers.
- $T$ is the hash table size (typically $2^{14}$ to $2^{24}$).
- Features are retrieved via trilinear interpolation of hashed vertex values.

**PyTorch Implementation:**
```python
import torch

def hash_fn(coords, hash_table_size):
    """
    Standard Instant-NGP spatial hashing
    """
    primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
    temp = coords * primes
    return torch.bitwise_xor(torch.bitwise_xor(temp[..., 0], temp[..., 1]), temp[..., 2]) % hash_table_size

# Usage in forward:
# 1. scaled_coords = coords * resolution
# 2. vertices = floor/ceil scaled_coords
# 3. features = hash_fn(vertices, T)
# 4. interpolate features
```

## 3. VM Decomposition (from TensoRF)

**Mathematical Formulation:**
Factorizes a 3D feature tensor $\mathcal{T}$ into a sum of Vector-Matrix outer products to reduce memory footprints while maintaining high resolution.
$$ \mathcal{T}_{i,j,k} = \sum_{r=1}^R \sum_{m \in \{X,Y,Z\}} \mathbf{v}_r^{(m)} \otimes \mathbf{M}_r^{(p_m)} $$
Specifically:
$$ \mathcal{T} = \sum_{r=1}^R \mathbf{v}_r^{(X)} \otimes \mathbf{M}_r^{(YZ)} + \mathbf{v}_r^{(Y)} \otimes \mathbf{M}_r^{(XZ)} + \mathbf{v}_r^{(Z)} \otimes \mathbf{M}_r^{(XY)} $$

**PyTorch Implementation:**
```python
import torch
import torch.nn.functional as F

def tensor_vm_encoding(coords, plane_coef, line_coef):
    """
    plane_coef: [3, R, Res, Res]
    line_coef: [3, R, Res, 1]
    """
    # Project coords onto planes (XY, XZ, YZ) and lines (Z, Y, X)
    plane_coords = torch.stack([coords[..., [0, 1]], coords[..., [0, 2]], coords[..., [1, 2]]])
    line_coords = torch.stack([coords[..., 2:3], coords[..., 1:2], coords[..., 0:1]])
    
    # Feature retrieval via grid sampling
    p_feat = F.grid_sample(plane_coef, plane_coords, align_corners=True)
    l_feat = F.grid_sample(line_coef, line_coords, align_corners=True)
    
    return (p_feat * l_feat).sum(dim=0) # Sum over the 3 orientations
```

## 4. Proposal Network (from Mip-NeRF 360)

**Mathematical Formulation (Inter-level Loss):**
The proposal network is a small MLP that predicts density to guide the sampling of the main NeRF. It is trained via an "inter-level" consistency loss:
$$ \mathcal{L}_{prop}(\mathbf{s}, \mathbf{w}, \mathbf{\hat{s}}, \mathbf{\hat{w}}) = \sum_{i} \frac{\max(0, w_i - \text{bound}(\mathbf{\hat{s}}, \mathbf{\hat{w}}, s_i, s_{i+1}))^2}{w_i + \epsilon} $$
Where:
- $\mathbf{w}$ are weights from the proposal network.
- $\mathbf{\hat{w}}$ are weights from the final "fine" NeRF.
- $\text{bound}(\cdot)$ is the sum of fine weights that overlap with the proposal interval $[s_i, s_{i+1})$.

**PyTorch Implementation:**
```python
import torch

def interlevel_loss(w_prop, s_prop, w_fine, s_fine):
    """
    Ensures proposal weights provide an upper bound for fine weights.
    w_outer is the sum of fine weights falling into each proposal interval
    """
    # Assuming w_outer is calculated from fine weights overlapping with proposal intervals
    # w_outer = calculate_overlapping_weights(w_fine, s_fine, s_prop)
    
    # Dummy placeholder for w_outer to demonstrate loss formula
    w_outer = w_fine 
    
    loss = torch.mean(torch.clamp(w_prop - w_outer, min=0)**2 / (w_prop + 1e-7))
    return loss
```
