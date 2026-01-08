import argparse
import numpy as np


argparser = argparse.ArgumentParser(description="Latent Health Analysis")

argparser.add_argument("logvar_mean", type=float, help="Mean of log-variance of posterior")
argparser.add_argument("mu_mean", type=float, help="Mean of the means of posterior")
argparser.add_argument("mu_std", type=float, help="Std of the means of posterior")

args = argparser.parse_args()


# Average posterior variance and std
avg_posterior_var = np.exp(args.logvar_mean)  # ≈ 0.322
avg_posterior_std = np.exp(args.logvar_mean / 2)  # ≈ 0.567

# Variance of the means (how spread out different samples are)
mu_variance = args.mu_std ** 2  # ≈ 0.449

# Total variance of z samples (should ≈ 1.0 for N(0,1) prior)
# By law of total variance: Var(z) = E[Var(z|x)] + Var(E[z|x])
total_z_variance = avg_posterior_var + mu_variance

print(f"Average posterior variance: {avg_posterior_var:.4f}")
print(f"Average posterior std: {avg_posterior_std:.4f}")
print(f"Variance of the means: {mu_variance:.4f}")
print(f"Total variance of z samples: {total_z_variance:.4f}")
