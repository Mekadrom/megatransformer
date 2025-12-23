#!/usr/bin/env python3
"""
Diagnostic script to identify why diffusion training loss plateaus at ~0.3.

Tests:
1. Gradient flow - Are gradients flowing through all layers?
2. Learning rate sensitivity - Does the model learn at different LRs?
3. Single sample memorization - Can the model overfit to ONE sample?
4. Output scale analysis - Is the model's output scale correct?
5. Loss landscape - Is the model in a local minimum?
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.image.diffusion import model_config_lookup, create_image_diffusion_model, tiny_image_diffusion_config


def test_gradient_flow(model, device="cuda"):
    """Test if gradients flow through all layers."""
    print("\n" + "="*60)
    print("TEST 1: Gradient Flow Analysis")
    print("="*60)

    model.train()
    model.to(device)

    # Create dummy input
    batch_size = 4
    latent_channels = 4
    latent_size = 32
    context_dim = 512

    x_0 = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    condition = torch.randn(batch_size, 512, context_dim, device=device)

    # Forward pass
    model.zero_grad()
    output, loss = model(x_0, condition=condition)

    # Backward pass
    loss.backward()

    # Check gradients for each parameter group
    print("\nGradient statistics per layer:")
    print("-" * 60)

    layers_with_zero_grad = []
    layers_with_small_grad = []
    layers_with_large_grad = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()

            status = ""
            if grad_norm < 1e-8:
                layers_with_zero_grad.append(name)
                status = " [ZERO GRAD!]"
            elif grad_norm < 1e-5:
                layers_with_small_grad.append(name)
                status = " [VERY SMALL]"
            elif grad_norm > 100:
                layers_with_large_grad.append(name)
                status = " [LARGE GRAD]"

            # Only print problematic layers
            if status:
                print(f"  {name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, std={grad_std:.6f}{status}")
        else:
            print(f"  {name}: NO GRADIENT!")
            layers_with_zero_grad.append(name)

    print(f"\nSummary:")
    print(f"  Layers with zero gradients: {len(layers_with_zero_grad)}")
    print(f"  Layers with very small gradients (<1e-5): {len(layers_with_small_grad)}")
    print(f"  Layers with large gradients (>100): {len(layers_with_large_grad)}")

    if layers_with_zero_grad:
        print(f"\nProblematic layers (zero grad): {layers_with_zero_grad[:5]}...")

    return len(layers_with_zero_grad) == 0


def test_single_sample_memorization(model, device="cuda", num_steps=500, lr=1e-4):
    """Test if the model can memorize a single sample."""
    print("\n" + "="*60)
    print(f"TEST 2: Single Sample Memorization (LR={lr})")
    print("="*60)

    model.train()
    model.to(device)

    # Create a fixed sample to memorize
    torch.manual_seed(42)
    latent_channels = 4
    latent_size = 32
    context_dim = 512

    x_0 = torch.randn(1, latent_channels, latent_size, latent_size, device=device)
    condition = torch.randn(1, 512, context_dim, device=device)

    print(f"Target x_0 stats: mean={x_0.mean():.4f}, std={x_0.std():.4f}")

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    losses = []
    for step in tqdm(range(num_steps), desc="Training"):
        optimizer.zero_grad()
        output, loss = model(x_0, condition=condition)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, output_std={output.std().item():.4f}")

    final_loss = losses[-1]
    min_loss = min(losses)

    print(f"\nResults:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Min loss: {min_loss:.4f} (at step {losses.index(min_loss)})")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss decreased by: {(losses[0] - min_loss) / losses[0] * 100:.1f}%")

    if min_loss < 0.1:
        print("  ✓ Model can memorize single sample - architecture is OK")
        return True
    elif min_loss < 0.2:
        print("  ~ Partial memorization - model has capacity but may need more steps or different LR")
        return True
    else:
        print("  ✗ Cannot memorize single sample - there may be an architectural issue")
        return False


def test_learning_rates(model_fn, device="cuda"):
    """Test different learning rates to find optimal range."""
    print("\n" + "="*60)
    print("TEST 3: Learning Rate Sensitivity")
    print("="*60)

    learning_rates = [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

    # Create fixed test data
    torch.manual_seed(42)
    latent_channels = 4
    latent_size = 32
    context_dim = 512
    batch_size = 8

    x_0 = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    condition = torch.randn(batch_size, 512, context_dim, device=device)

    results = {}

    for lr in learning_rates:
        print(f"\nTesting LR={lr}...")

        # Create fresh model
        model = model_fn()
        model.train()
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

        losses = []
        for step in range(100):
            optimizer.zero_grad()
            output, loss = model(x_0, condition=condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        results[lr] = {
            'initial': losses[0],
            'final': losses[-1],
            'min': min(losses),
            'improvement': (losses[0] - min(losses)) / losses[0] * 100
        }

        print(f"  Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}, Min: {min(losses):.4f}, Improvement: {results[lr]['improvement']:.1f}%")

    print("\nSummary:")
    print("-" * 60)
    best_lr = max(results.keys(), key=lambda lr: results[lr]['improvement'])
    print(f"Best LR: {best_lr} with {results[best_lr]['improvement']:.1f}% improvement")

    return best_lr


def test_output_scale(model, device="cuda"):
    """Test if model output scale matches target scale at initialization and after training."""
    print("\n" + "="*60)
    print("TEST 4: Output Scale Analysis")
    print("="*60)

    model.eval()
    model.to(device)

    batch_size = 32
    latent_channels = 4
    latent_size = 32

    # Test at initialization
    with torch.no_grad():
        # Create inputs with various noise levels
        for t_value in [0, 250, 500, 750, 999]:
            x_noisy = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.long)

            output = model.unet(x_noisy, t, condition=None)

            print(f"  t={t_value}: input_std={x_noisy.std():.4f}, output_std={output.std():.4f}, ratio={output.std()/x_noisy.std():.4f}")

    # For a well-initialized noise predictor, output_std should be close to 1.0
    print("\nExpected: output_std ≈ 1.0 (for epsilon prediction) or ≈ target_v_std (for v prediction)")


def test_target_statistics(model, device="cuda"):
    """Analyze target statistics for v-prediction vs epsilon."""
    print("\n" + "="*60)
    print("TEST 5: Target Statistics Analysis")
    print("="*60)

    model.eval()
    model.to(device)

    batch_size = 256  # Large batch for accurate statistics
    latent_channels = 4
    latent_size = 32

    x_start = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    noise = torch.randn_like(x_start)

    print(f"x_start: mean={x_start.mean():.4f}, std={x_start.std():.4f}")
    print(f"noise: mean={noise.mean():.4f}, std={noise.std():.4f}")

    # Test v-target at different timesteps
    print("\nV-target statistics at different timesteps:")
    print("-" * 60)

    for t_value in [0, 100, 250, 500, 750, 900, 999]:
        t = torch.full((batch_size,), t_value, device=device, dtype=torch.long)

        v_target = model.get_v_target(x_start, noise, t)

        # Get alpha values for reference
        sqrt_alpha = model._extract(model.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1m_alpha = model._extract(model.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        print(f"  t={t_value}: v_std={v_target.std():.4f}, sqrt_alpha={sqrt_alpha[0,0,0,0]:.4f}, sqrt_1m_alpha={sqrt_1m_alpha[0,0,0,0]:.4f}")

    print("\nNote: v_target = sqrt_alpha * noise - sqrt_1m_alpha * x_start")
    print("At t=0 (low noise): v ≈ -x_start, std ≈ x_start_std")
    print("At t=999 (high noise): v ≈ noise, std ≈ 1.0")


def test_loss_vs_random_predictor(model, device="cuda"):
    """Compare model loss to baseline random predictor."""
    print("\n" + "="*60)
    print("TEST 6: Loss vs Random Predictor Baseline")
    print("="*60)

    model.eval()
    model.to(device)

    batch_size = 64
    latent_channels = 4
    latent_size = 32
    context_dim = 512

    x_0 = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    condition = torch.randn(batch_size, 512, context_dim, device=device)

    with torch.no_grad():
        # Get actual model loss
        output, loss = model(x_0, condition=condition)
        model_loss = loss.item()

        # Calculate baseline losses
        # For v-prediction, target is v = sqrt_alpha * noise - sqrt_1m_alpha * x_start
        # If predicting zeros: loss = E[v^2] ≈ 1.0 (depends on timestep distribution)
        # If predicting random N(0,1): loss = E[(v - random)^2] = Var(v) + Var(random) ≈ 2.0

        # Estimate actual target variance
        noise = torch.randn_like(x_0)
        t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)

        if model.prediction_type == "v":
            target = model.get_v_target(x_0, noise, t)
        else:
            target = noise

        target_var = target.var().item()

        # Loss if predicting zeros
        zero_pred_loss = (target ** 2).mean().item()

        # Loss if predicting random noise
        random_pred = torch.randn_like(target)
        random_pred_loss = F.mse_loss(random_pred, target).item()

    print(f"Model loss: {model_loss:.4f}")
    print(f"Zero predictor loss: {zero_pred_loss:.4f}")
    print(f"Random predictor loss: {random_pred_loss:.4f}")
    print(f"Target variance: {target_var:.4f}")

    if model_loss < zero_pred_loss * 0.9:
        print("\n✓ Model is better than zero predictor - it's learning something")
    else:
        print("\n✗ Model is not better than zero predictor - may be stuck")

    improvement = (zero_pred_loss - model_loss) / zero_pred_loss * 100
    print(f"Improvement over zero baseline: {improvement:.1f}%")


def test_gradient_magnitudes_per_block(model, device="cuda"):
    """Analyze gradient magnitudes through different blocks."""
    print("\n" + "="*60)
    print("TEST 7: Gradient Magnitudes Per Block")
    print("="*60)

    model.train()
    model.to(device)

    batch_size = 8
    latent_channels = 4
    latent_size = 32
    context_dim = 512

    x_0 = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    condition = torch.randn(batch_size, 512, context_dim, device=device)

    model.zero_grad()
    output, loss = model(x_0, condition=condition)
    loss.backward()

    # Group parameters by block
    block_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Extract block name
            parts = name.split('.')
            if 'down_blocks' in name:
                idx = name.find('down_blocks')
                block_name = name[idx:].split('.')[0] + '.' + name[idx:].split('.')[1]
            elif 'up_blocks' in name:
                idx = name.find('up_blocks')
                block_name = name[idx:].split('.')[0] + '.' + name[idx:].split('.')[1]
            elif 'middle' in name:
                block_name = 'middle'
            elif 'time_' in name:
                block_name = 'time_embedding'
            elif 'init_conv' in name:
                block_name = 'init_conv'
            elif 'final' in name:
                block_name = 'final'
            else:
                block_name = 'other'

            if block_name not in block_grads:
                block_grads[block_name] = []
            block_grads[block_name].append(param.grad.norm().item())

    print("\nGradient norms per block:")
    print("-" * 60)
    for block_name, grad_norms in sorted(block_grads.items()):
        mean_grad = np.mean(grad_norms)
        max_grad = np.max(grad_norms)
        print(f"  {block_name}: mean={mean_grad:.6f}, max={max_grad:.6f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model factory for fresh models
    def create_model():
        return model_config_lookup["balanced_small_image_diffusion"](
            latent_channels=4,
            num_timesteps=1000,
            sampling_timesteps=20,
            betas_schedule="cosine",
            context_dim=512,
            normalize=False,  # For latent diffusion
            min_snr_loss_weight=False,  # Disabled for testing
            cfg_dropout_prob=0.0,  # Disable for testing
            zero_terminal_snr=False,  # Disable for testing
            offset_noise_strength=0.0,  # Disable for testing
            timestep_sampling="uniform",  # Simple for testing
        )

    # Create main model for tests
    model = create_model()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run all tests
    test_gradient_flow(model, device)
    test_output_scale(model, device)
    test_target_statistics(model, device)
    test_loss_vs_random_predictor(model, device)
    test_gradient_magnitudes_per_block(model, device)

    # Test single sample memorization with different LRs
    print("\n" + "="*60)
    print("SINGLE SAMPLE MEMORIZATION TESTS")
    print("="*60)

    for lr in [1e-3, 1e-4, 1e-5]:
        model = create_model()
        test_single_sample_memorization(model, device, num_steps=300, lr=lr)

    # Learning rate sensitivity test
    test_learning_rates(create_model, device)


if __name__ == "__main__":
    main()