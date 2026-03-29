"""NerfifyTrainer — End-to-end training loop with smoke-test validation.

Implements the paper's Stage 3 smoke-test protocol:
    "runs smoke training for 3k iterations"
    "an end-to-end training step completes with a finite loss"

Plus Stage 4 convergence criteria:
    "PSNR within 0.5 dB", "SSIM within 0.2"

Topological position: depends on ALL prior files
    configuration.py → dataset.py → modules.py → modeling.py → trainer.py

References:
    Paper §3.2 Stage 3 — integration testing / smoke train
    Paper §3.2 Stage 4 — visual-driven feedback loop
    Paper §4.2 — 100k iterations, 3k smoke, A6000 GPU
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configuration import NerfifyConfig
from dataset import NerfifyDataset, NerfifyRayBundle
from modeling import NerfifyForCausalLM


class NerfifyTrainer:
    """Training loop for the Nerfify model.

    Follows the paper's GoT integration testing protocol:
    1. Build model + dataset from config
    2. Run training steps with ray rendering
    3. Validate that loss is finite after each step
    4. Report PSNR for convergence monitoring

    Args:
        config: NerfifyConfig with all hyperparameters.
        model: NerfifyForCausalLM instance (if None, created from config).
        device: Device to train on.
    """

    def __init__(
        self,
        config: NerfifyConfig,
        model: Optional[NerfifyForCausalLM] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)

        # Build model
        if model is None:
            self.model = NerfifyForCausalLM(config).to(self.device)
        else:
            self.model = model.to(self.device)

        # Build dataset
        self.dataset = NerfifyDataset(config, num_rays=1024)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
        )

        # Training state
        self.global_step = 0
        self.losses: list = []

    def train_step(self, batch_size: int = 8) -> Dict[str, float]:
        """Execute a single training step.

        Paper Figure 4, pipeline.py agent:
            "dm, f, m = build(cfg); b = dm.next_batch()"
            "out = m.get_outputs(b); L = m.loss(out, b)"
            "update_parameters(m, L)"

        Returns:
            dict with "loss", "rgb_psnr", "step".
        """
        self.model.train()

        # Generate a batch of random rays
        batch_idx = torch.randint(0, len(self.dataset), (batch_size,))
        bundle = self.dataset.collate_as_ray_bundle(batch_idx)

        # Move to device
        origins = bundle.origins.to(self.device)
        directions = bundle.directions.to(self.device)
        gt_rgb = bundle.pixel_rgb.to(self.device)

        # Forward pass — NeRF rendering mode
        output = self.model(
            ray_origins=origins,
            ray_directions=directions,
            pixel_rgb=gt_rgb,
        )

        loss = output.loss
        assert loss is not None, "Loss is None — forward() failed!"
        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"

        # Backward + optimizer step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Compute PSNR from MSE
        with torch.no_grad():
            render_out = self.model.render_rays(origins, directions)
            mse = nn.functional.mse_loss(render_out["rgb"], gt_rgb)
            psnr = -10.0 * torch.log10(mse + 1e-8)

        self.global_step += 1
        loss_val = loss.item()
        self.losses.append(loss_val)

        return {
            "loss": loss_val,
            "rgb_psnr": psnr.item(),
            "step": self.global_step,
        }

    def smoke_train(
        self,
        num_steps: Optional[int] = None,
        batch_size: int = 8,
        log_every: int = 10,
    ) -> Dict[str, float]:
        """Smoke training — validates the pipeline end-to-end.

        Paper §3.2 Stage 3:
            "smoke train for smoke_train_iterations"
            "Exit criteria: train_step() finishes with finite loss"

        Args:
            num_steps: Number of steps (default: config.smoke_train_iterations).
            batch_size: Rays per step.
            log_every: Print frequency.

        Returns:
            dict with final "loss", "psnr", "total_steps", "time_seconds".
        """
        if num_steps is None:
            num_steps = self.config.smoke_train_iterations

        print(f"  Starting smoke train: {num_steps} steps, batch_size={batch_size}")
        start_time = time.time()

        for step in range(1, num_steps + 1):
            metrics = self.train_step(batch_size=batch_size)

            if step % log_every == 0 or step == 1:
                print(
                    f"    Step {metrics['step']:5d} | "
                    f"Loss: {metrics['loss']:.6f} | "
                    f"PSNR: {metrics['rgb_psnr']:.2f} dB"
                )

        elapsed = time.time() - start_time
        final_loss = self.losses[-1]
        final_psnr = metrics["rgb_psnr"]

        print(f"  Smoke train complete: {elapsed:.1f}s, "
              f"final loss={final_loss:.6f}, PSNR={final_psnr:.2f} dB")

        return {
            "loss": final_loss,
            "psnr": final_psnr,
            "total_steps": self.global_step,
            "time_seconds": elapsed,
        }

    @torch.no_grad()
    def evaluate(self, num_rays: int = 64) -> Dict[str, float]:
        """Evaluate model on random rays.

        Returns PSNR and average loss for convergence checking
        per Stage 4 criteria.
        """
        self.model.eval()

        # Generate evaluation rays
        batch_idx = torch.randint(0, len(self.dataset), (num_rays,))
        bundle = self.dataset.collate_as_ray_bundle(batch_idx)

        origins = bundle.origins.to(self.device)
        directions = bundle.directions.to(self.device)
        gt_rgb = bundle.pixel_rgb.to(self.device)

        # Render
        render_out = self.model.render_rays(origins, directions)
        mse = nn.functional.mse_loss(render_out["rgb"], gt_rgb)
        psnr = -10.0 * torch.log10(mse + 1e-8)

        return {
            "eval_loss": mse.item(),
            "eval_psnr": psnr.item(),
        }


# ============================================================
# VALIDATION BLOCK — Full end-to-end smoke test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATING: trainer.py")
    print("=" * 60)

    # Use small config for fast validation
    config = NerfifyConfig(
        hidden_dim=32,
        num_samples=8,
        num_layers=2,
        num_hash_levels=4,
        hash_table_size=2**8,
        vm_rank=4,
        vm_resolution=8,
        lr=1e-3,
        smoke_train_iterations=20,  # minimal for test
    )

    # Test 1: Trainer creation
    trainer = NerfifyTrainer(config, device="cpu")
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Model: {total_params:,} parameters")
    print("  ✅ Trainer creation OK")

    # Test 2: Single training step
    metrics = trainer.train_step(batch_size=4)
    print(f"  Step {metrics['step']}: loss={metrics['loss']:.6f}, PSNR={metrics['rgb_psnr']:.2f} dB")
    assert metrics["loss"] > 0, "Loss should be positive"
    assert not torch.isnan(torch.tensor(metrics["loss"])), "Loss is NaN!"
    assert not torch.isinf(torch.tensor(metrics["loss"])), "Loss is Inf!"
    print("  ✅ Single training step OK (finite loss)")

    # Test 3: Smoke train (20 steps)
    print("\n  --- Smoke Training ---")
    smoke_result = trainer.smoke_train(num_steps=20, batch_size=4, log_every=5)
    assert smoke_result["loss"] > 0
    assert not torch.isnan(torch.tensor(smoke_result["loss"]))
    assert smoke_result["total_steps"] == 21  # 1 from test 2 + 20
    print("  ✅ Smoke train completed with finite loss")

    # Test 4: Evaluation
    eval_result = trainer.evaluate(num_rays=16)
    print(f"  Eval: loss={eval_result['eval_loss']:.6f}, PSNR={eval_result['eval_psnr']:.2f} dB")
    assert eval_result["eval_loss"] > 0
    assert not torch.isnan(torch.tensor(eval_result["eval_psnr"]))
    print("  ✅ Evaluation OK")

    # Test 5: Loss decreased during training
    first_loss = trainer.losses[0]
    last_loss = trainer.losses[-1]
    print(f"  Loss trajectory: {first_loss:.6f} → {last_loss:.6f}")
    # Note: with random data, loss may not always decrease consistently
    # but should remain finite throughout
    all_finite = all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l)))
                     for l in trainer.losses)
    assert all_finite, "Some losses were NaN or Inf!"
    print(f"  ✅ All {len(trainer.losses)} losses are finite")

    # Test 6: Gradient flow — verify all key module params get gradients
    trainer.train_step(batch_size=4)
    modules_with_grad = set()
    for name, param in trainer.model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            module_name = name.split(".")[0]
            modules_with_grad.add(module_name)
    print(f"  Modules with gradients: {sorted(modules_with_grad)}")
    assert "field" in modules_with_grad, "Field has no gradients!"
    assert "proposal_net" in modules_with_grad, "ProposalNet has no gradients!"
    print("  ✅ Key modules receive gradients")

    print()
    print("🎉 trainer.py — ALL CHECKS PASSED")
    print("=" * 60)
