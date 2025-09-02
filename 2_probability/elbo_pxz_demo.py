#!/usr/bin/env python3
"""
ELBO & p(x,z) Demo (GMM + VAE-like)
-----------------------------------

This script shows:
- How to compute p(x,z) = p(z) p(x|z) for a Gaussian Mixture Model (GMM)
  with discrete z (easy to evaluate exactly).
- How to compute log p(x,z) for a VAE-like toy model with
  p(z)=N(0,I), p(x|z)=N(f_theta(z), sigma^2 I); evaluating at a given z is easy.
- How to estimate the ELBO with Monte Carlo:
      ELBO(q) = E_q[ log p(x,z) - log q(z|x) ]
  without ever needing p(x) or the true posterior.

No 3rd-party libraries required beyond numpy.
"""

import math
import numpy as np
from typing import Tuple, Callable, Dict

# ---------------------------
# Utility: Gaussian logpdf(s)
# ---------------------------

def logpdf_standard_normal(z: np.ndarray) -> float:
    """
    log N(z; 0, I) for vector z.
    """
    d = z.size
    return -0.5 * (d * math.log(2 * math.pi) + np.dot(z, z))


def logpdf_diag_normal(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """
    log N(x; mean, diag(var))
    var must be positive; we assume it's well-formed.
    """
    d = x.size
    diff = x - mean
    return -0.5 * (d * math.log(2 * math.pi) + np.sum(np.log(var)) + np.sum((diff * diff) / var))


def logpdf_full_normal(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    log N(x; mean, cov) for full (positive definite) covariance.
    """
    d = x.size
    diff = (x - mean).reshape(-1, 1)
    # Cholesky for stability
    L = np.linalg.cholesky(cov)
    # Solve L y = diff
    y = np.linalg.solve(L, diff)
    quad = float(np.dot(y.T, y))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * math.log(2 * math.pi) + logdet + quad)


# ---------------------------
# Example 1: GMM
# ---------------------------

class GMM:
    def __init__(self, pis: np.ndarray, mus: np.ndarray, covs: np.ndarray):
        """
        pis: (K,) mixture weights (sum to 1)
        mus: (K, D)
        covs: (K, D, D) full covariances (PD)
        """
        self.pis = pis
        self.mus = mus
        self.covs = covs
        self.K = pis.shape[0]
        self.D = mus.shape[1]

    def log_pz(self, z_k: int) -> float:
        return math.log(self.pis[z_k])

    def log_px_given_z(self, x: np.ndarray, z_k: int) -> float:
        return logpdf_full_normal(x, self.mus[z_k], self.covs[z_k])

    def log_pxz(self, x: np.ndarray, z_k: int) -> float:
        # log p(x,z=k) = log p(z=k) + log p(x|z=k)
        return self.log_pz(z_k) + self.log_px_given_z(x, z_k)

    def log_px(self, x: np.ndarray) -> float:
        # For discrete z, the marginal is a sum over K components
        # log sum exp in a stable way
        logs = np.array([self.log_pxz(x, k) for k in range(self.K)])
        m = np.max(logs)
        return m + math.log(np.sum(np.exp(logs - m)))


# ---------------------------
# Example 2: VAE-like toy model
# ---------------------------

class VAEToy:
    def __init__(self, D_z: int, D_x: int, sigma2: float, W: np.ndarray, b: np.ndarray):
        """
        p(z) = N(0, I_{D_z})
        p(x|z) = N(f_theta(z), sigma^2 I_{D_x}), where f_theta(z) = W z + b (linear for simplicity).
        - W: (D_x, D_z), b: (D_x,)
        """
        self.D_z = D_z
        self.D_x = D_x
        self.sigma2 = sigma2
        self.W = W
        self.b = b

    def f_theta(self, z: np.ndarray) -> np.ndarray:
        return self.W @ z + self.b

    def log_pz(self, z: np.ndarray) -> float:
        return logpdf_standard_normal(z)

    def log_px_given_z(self, x: np.ndarray, z: np.ndarray) -> float:
        mean = self.f_theta(z)
        var = np.full(self.D_x, self.sigma2)
        return logpdf_diag_normal(x, mean, var)

    def log_pxz(self, x: np.ndarray, z: np.ndarray) -> float:
        # log p(x,z) = log p(z) + log p(x|z)
        return self.log_pz(z) + self.log_px_given_z(x, z)


# ---------------------------------------
# Variational family q(z|x): diagonal Gaussian
# ---------------------------------------

def sample_q_diag_normal(mu: np.ndarray, logvar: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Reparameterized samples: z = mu + exp(0.5*logvar) * eps, eps ~ N(0,I).
    Returns (n_samples, D).
    """
    D = mu.size
    eps = rng.normal(size=(n_samples, D))
    std = np.exp(0.5 * logvar)
    return mu + std * eps


def log_q_diag_normal(z: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> float:
    var = np.exp(logvar)
    return logpdf_diag_normal(z, mu, var)


# ---------------------------
# Monte Carlo ELBO
# ---------------------------

def elbo_mc(
    x: np.ndarray,
    log_p_xz: Callable[[np.ndarray, np.ndarray], float],
    sample_q: Callable[[int], np.ndarray],
    log_q: Callable[[np.ndarray], float],
    n_samples: int = 100,
    rng: np.random.Generator = None
) -> float:
    """
    Estimates ELBO = E_q[ log p(x,z) - log q(z|x) ] with Monte Carlo.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Draw samples from q(z|x)
    zs = sample_q(n_samples)

    vals = []
    for z in zs:
        vals.append(log_p_xz(x, z) - log_q(z))
    return float(np.mean(vals))


# ---------------------------
# Demos
# ---------------------------

def demo_gmm():
    print("=== Demo: GMM p(x,z) vs p(x) ===")
    rng = np.random.default_rng(42)
    K, D = 3, 2
    pis = np.array([0.5, 0.3, 0.2])
    mus = np.array([[0.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 3.0]])
    covs = np.stack([np.eye(D) for _ in range(K)], axis=0)

    gmm = GMM(pis, mus, covs)

    # Pick an x and a particular component z=k
    x = np.array([0.5, -0.2])
    z_k = 1

    print(f"log p(x,z={z_k}) = {gmm.log_pxz(x, z_k):.4f}  (exact, easy: log pi_k + log N)")
    print(f"log p(x)         = {gmm.log_px(x):.4f}  (exact here via sum over K)")

    # Note: For large/continuous z, the exact marginal log p(x) is no longer a simple finite sum.


def demo_vae_like():
    print("\n=== Demo: VAE-like p(x,z) and MC-ELBO ===")
    rng = np.random.default_rng(123)
    D_z, D_x = 4, 3
    sigma2 = 0.25
    W = rng.normal(scale=0.5, size=(D_x, D_z))
    b = rng.normal(scale=0.1, size=(D_x,))

    model = VAEToy(D_z, D_x, sigma2, W, b)

    # One observed x
    x = rng.normal(size=(D_x,))

    # Variational parameters of q(z|x) = N(mu, diag(exp(logvar)))
    mu = rng.normal(size=(D_z,))
    logvar = np.full(D_z, -0.5)  # std ~ 0.61

    # Define closures for ELBO estimator
    def log_p_xz_closure(x_in, z_in):
        return model.log_pxz(x_in, z_in)

    def sampler_q(n):
        return sample_q_diag_normal(mu, logvar, n, rng)

    def log_q_closure(z_in):
        return log_q_diag_normal(z_in, mu, logvar)

    # MC estimate
    est = elbo_mc(x, log_p_xz_closure, sampler_q, log_q_closure, n_samples=500, rng=rng)

    # Also show one example of "computing p(x,z) at a single z" explicitly
    z_single = sampler_q(1)[0]
    log_pxz_single = model.log_pxz(x, z_single)
    log_q_single = log_q_closure(z_single)

    print(f"log p(x,z) at a sampled z: {log_pxz_single:.4f} (computable: log p(z) + log p(x|z))")
    print(f"log q(z|x) at that z:      {log_q_single:.4f} (computable: chosen variational family)")
    print(f"MC-ELBO estimate (500 samples): {est:.4f}")
    print("\nNote: We never computed p(x) or the true posterior. ELBO uses only log p(x,z) and log q(z|x).")


if __name__ == '__main__':
    demo_gmm()
    demo_vae_like()
