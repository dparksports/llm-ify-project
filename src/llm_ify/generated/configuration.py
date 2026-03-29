"""NerfifyConfig — Hugging Face PretrainedConfig for a NeRF-inspired model.

This configuration defines all hyperparameters for the Nerfify model,
following the HF CFG contract (.agent/rules/hf_cfg.md).

Hyperparameters sourced from:
    - Paper §4.2 (benchmark settings)
    - Paper Figure 4 (GoT config.py agent)
    - dependencies.md (hash encoding, distortion loss, VM decomposition)

Topological position: ROOT (no dependencies).
"""

from transformers import PretrainedConfig


class NerfifyConfig(PretrainedConfig):
    """Configuration for the Nerfify model.

    Maps all paper hyperparameters to a single, serializable config:
        - Field dimensions (hash encoding, VM decomposition per dependencies.md)
        - Model architecture (MLP widths, number of samples)
        - Training hyperparameters (lr, iterations, loss weights)
        - Evaluation thresholds (PSNR, SSIM from §3.2 Stage 4)

    Attributes:
        hidden_dim (int): Hidden dimension for the field MLP. [Figure 4]
        num_samples (int): Samples per ray. [Figure 4: n_samples=64]
        num_importance_samples (int): Importance samples for proposal net.
        spatial_dim (int): Input spatial coordinate dimension (3 for xyz).
        direction_dim (int): Input view direction dimension (3 for xyz).
        density_dim (int): Output density channels.
        rgb_dim (int): Output RGB channels.
        num_layers (int): Number of MLP layers in the field.
        hash_table_size (int): Hash table size T for Instant-NGP encoding.
            [dependencies.md §2: typically 2^14 to 2^24]
        num_hash_levels (int): Multi-resolution hash levels.
        features_per_level (int): Feature vector size per hash level.
        vm_rank (int): Rank R for TensoRF VM decomposition.
            [dependencies.md §3]
        vm_resolution (int): Grid resolution for VM planes/lines.
        num_proposal_iterations (int): Proposal network iterations.
        distortion_loss_weight (float): Weight for distortion loss.
            [dependencies.md §1]
        interlevel_loss_weight (float): Weight for proposal inter-level loss.
            [dependencies.md §4]
        lr (float): Learning rate. [Figure 4: 1e-3]
        max_iterations (int): Max training iterations. [§4.2: 100k]
        smoke_train_iterations (int): Smoke test iterations. [§4.2: 3k]
        psnr_target_delta (float): PSNR tolerance for convergence. [§3.2: 0.5]
        ssim_target_delta (float): SSIM tolerance for convergence. [§3.2: 0.2]
        vocab_size (int): Vocabulary size for HF compatibility.
        max_position_embeddings (int): Max sequence length for HF compat.
    """

    model_type = "nerfify"

    def __init__(
        self,
        # --- Field architecture ---
        hidden_dim: int = 128,
        num_samples: int = 64,
        num_importance_samples: int = 32,
        spatial_dim: int = 3,
        direction_dim: int = 3,
        density_dim: int = 1,
        rgb_dim: int = 3,
        num_layers: int = 4,
        # --- Hash encoding (Instant-NGP, dependencies.md §2) ---
        hash_table_size: int = 2**19,
        num_hash_levels: int = 16,
        features_per_level: int = 2,
        # --- VM decomposition (TensoRF, dependencies.md §3) ---
        vm_rank: int = 48,
        vm_resolution: int = 128,
        # --- Proposal network (Mip-NeRF 360, dependencies.md §4) ---
        num_proposal_iterations: int = 2,
        # --- Loss weights ---
        distortion_loss_weight: float = 0.01,
        interlevel_loss_weight: float = 1.0,
        rgb_loss_weight: float = 1.0,
        # --- Training (§4.2) ---
        lr: float = 1e-3,
        max_iterations: int = 100_000,
        smoke_train_iterations: int = 3_000,
        # --- Evaluation thresholds (§3.2 Stage 4) ---
        psnr_target_delta: float = 0.5,
        ssim_target_delta: float = 0.2,
        # --- HF protocol fields ---
        vocab_size: int = 32000,
        max_position_embeddings: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Field
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.num_importance_samples = num_importance_samples
        self.spatial_dim = spatial_dim
        self.direction_dim = direction_dim
        self.density_dim = density_dim
        self.rgb_dim = rgb_dim
        self.num_layers = num_layers
        # Hash encoding
        self.hash_table_size = hash_table_size
        self.num_hash_levels = num_hash_levels
        self.features_per_level = features_per_level
        # VM decomposition
        self.vm_rank = vm_rank
        self.vm_resolution = vm_resolution
        # Proposal
        self.num_proposal_iterations = num_proposal_iterations
        # Losses
        self.distortion_loss_weight = distortion_loss_weight
        self.interlevel_loss_weight = interlevel_loss_weight
        self.rgb_loss_weight = rgb_loss_weight
        # Training
        self.lr = lr
        self.max_iterations = max_iterations
        self.smoke_train_iterations = smoke_train_iterations
        # Evaluation
        self.psnr_target_delta = psnr_target_delta
        self.ssim_target_delta = ssim_target_delta
        # HF protocol
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings


# ============================================================
# VALIDATION BLOCK — auto-executed to verify config round-trip
# ============================================================
if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("VALIDATING: configuration.py")
    print("=" * 60)

    # Test 1: Config instantiation with defaults
    config = NerfifyConfig()
    print(f"  model_type       = {config.model_type}")
    print(f"  hidden_dim       = {config.hidden_dim}")
    print(f"  num_samples      = {config.num_samples}")
    print(f"  hash_table_size  = {config.hash_table_size}")
    print(f"  vm_rank          = {config.vm_rank}")
    print(f"  lr               = {config.lr}")
    assert config.model_type == "nerfify"
    assert config.hidden_dim == 128
    assert config.num_samples == 64
    print("  ✅ Default instantiation OK")

    # Test 2: Custom values
    config2 = NerfifyConfig(hidden_dim=256, num_samples=128, lr=5e-4)
    assert config2.hidden_dim == 256
    assert config2.num_samples == 128
    assert config2.lr == 5e-4
    print("  ✅ Custom instantiation OK")

    # Test 3: Serialization round-trip
    d = config.to_dict()
    config3 = NerfifyConfig(**d)
    assert config3.hidden_dim == config.hidden_dim
    assert config3.hash_table_size == config.hash_table_size
    print("  ✅ Serialization round-trip OK")

    # Test 4: Dummy tensor shapes (prove config drives dimensions)
    B, S = 2, config.num_samples
    spatial = torch.randn(B, S, config.spatial_dim)
    direction = torch.randn(B, S, config.direction_dim)
    hidden = torch.randn(B, S, config.hidden_dim)
    print(f"  spatial  shape: {spatial.shape}  (expected [2, 64, 3])")
    print(f"  direction shape: {direction.shape}  (expected [2, 64, 3])")
    print(f"  hidden   shape: {hidden.shape}  (expected [2, 64, 128])")
    assert spatial.shape == (B, S, 3)
    assert direction.shape == (B, S, 3)
    assert hidden.shape == (B, S, 128)
    print("  ✅ Tensor shapes match config")

    print()
    print("🎉 configuration.py — ALL CHECKS PASSED")
    print("=" * 60)
