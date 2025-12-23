#!/usr/bin/env python3
"""
Generate graphs for all non-parameterized activation functions supported in megatransformer.
Saves individual plots and a combined comparison plot to logs/ folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# torch.nn.functional not needed - using nn.Module activations

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Define x range for plotting
x = torch.linspace(-5, 5, 1000)
x_np = x.numpy()

# Non-parameterized activation functions from megatransformer_utils.get_activation_type
activations = {
    "ReLU": nn.ReLU(),
    "GELU": nn.GELU(),
    "ELU": nn.ELU(),
    "SELU": nn.SELU(),
    "LeakyReLU": nn.LeakyReLU(),
    "SiLU (Swish)": nn.SiLU(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "Identity": nn.Identity(),
}

# Color palette
colors = plt.cm.tab10(np.linspace(0, 1, len(activations)))

# Generate individual plots
for i, (name, activation) in enumerate(activations.items()):
    fig, ax = plt.subplots(figsize=(8, 6))

    with torch.no_grad():
        y = activation(x).numpy()

    ax.plot(x_np, y, color=colors[i], linewidth=2.5, label=name)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'{name} Activation Function', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)

    # Set appropriate y limits based on activation
    if name in ["Sigmoid"]:
        ax.set_ylim(-0.2, 1.2)
    elif name in ["Tanh"]:
        ax.set_ylim(-1.5, 1.5)
    elif name in ["ELU", "SELU"]:
        ax.set_ylim(-2, 5)
    else:
        ax.set_ylim(-2, 5)

    filename = f"logs/activation_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Generate combined comparison plot
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()

for i, (name, activation) in enumerate(activations.items()):
    ax = axes[i]

    with torch.no_grad():
        y = activation(x).numpy()

    ax.plot(x_np, y, color=colors[i], linewidth=2.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)

    if name in ["Sigmoid"]:
        ax.set_ylim(-0.2, 1.2)
    elif name in ["Tanh"]:
        ax.set_ylim(-1.5, 1.5)
    else:
        ax.set_ylim(-2, 5)

fig.suptitle('Non-Parameterized Activation Functions (MegaTransformer)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("logs/activations_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: logs/activations_comparison.png")

# Generate overlay plot (all on one axis)
fig, ax = plt.subplots(figsize=(12, 8))

for i, (name, activation) in enumerate(activations.items()):
    with torch.no_grad():
        y = activation(x).numpy()
    ax.plot(x_np, y, color=colors[i], linewidth=2, label=name)

ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('All Non-Parameterized Activation Functions Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 5)

plt.tight_layout()
plt.savefig("logs/activations_overlay.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: logs/activations_overlay.png")

# Generate derivative comparison plot
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()

for i, (name, activation) in enumerate(activations.items()):
    ax = axes[i]

    x_grad = x.clone().requires_grad_(True)
    y = activation(x_grad)

    # Compute gradient
    grad = torch.autograd.grad(y.sum(), x_grad)[0].detach().numpy()

    ax.plot(x_np, grad, color=colors[i], linewidth=2.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_title(f"{name} (derivative)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, 2)

fig.suptitle('Activation Function Derivatives (MegaTransformer)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("logs/activations_derivatives.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: logs/activations_derivatives.png")

print("\nAll activation function plots generated successfully!")