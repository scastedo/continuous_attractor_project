#!/usr/bin/env python3
"""Toy spectral model: top-hat CANN J, bump-derived D, and SNR.

Only numpy + matplotlib. No time-series simulation.
"""

import numpy as np
import matplotlib.pyplot as plt


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def build_top_hat_j(theta, width, height):
    """Top-hat connectivity on a ring with 1/N scaling."""
    n = theta.size
    dtheta = wrap_to_pi(theta[:, None] - theta[None, :])
    mask = (np.abs(dtheta) <= width).astype(float)
    return (height / n) * mask


def von_mises_bump(theta, center, kappa):
    x = np.exp(kappa * np.cos(theta - center))
    x /= x.max()
    return x


def build_rho_profile(theta, center, kappa, rho_floor, rho_peak):
    bump = von_mises_bump(theta, center, kappa)
    rho = rho_floor + (rho_peak - rho_floor) * bump
    return rho


def delta_x(theta, base_center, dtheta, kappa, rho_floor, rho_peak):
    x1 = build_rho_profile(theta, base_center, kappa, rho_floor, rho_peak)
    x2 = build_rho_profile(theta, base_center + dtheta, kappa, rho_floor, rho_peak)
    dx = x1 - x2
    dx -= dx.mean()
    return dx


def gamma_eigs_from_lambda_j(lam_j, w, r, d):
    denom = 1.0 - (r * w * d) * lam_j
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("Unstable: 1 - r*w*d*lambda_J hits 0. Reduce gain or scale J down.")
    return (r * d) / denom


def snr_curve(theta, evecs, lam_g, w, sigma_gamma, alpha,
              base_center, kappa, rho_floor, rho_peak,
              dtheta_grid, normalize_dx):
    # In the eigenbasis, Sigma is diagonal, so SNR is a weighted sum.
    lam_sigma = (w * sigma_gamma) ** 2 * lam_g**2 + alpha * lam_g

    snr = np.zeros_like(dtheta_grid)
    for i, dth in enumerate(dtheta_grid):
        dx = delta_x(theta, base_center, dth, kappa, rho_floor, rho_peak)
        if normalize_dx:
            dx /= np.linalg.norm(dx)
        dx_k = evecs.T @ dx
        dmu_k = w * lam_g * dx_k
        snr[i] = np.sum((dmu_k**2) / lam_sigma)
    return snr


def fast_noise_metrics(theta_vals, snr_ctr, snr_fr, small_pct, large_pct):
    small_idx = theta_vals <= np.percentile(theta_vals, small_pct)
    large_idx = theta_vals >= np.percentile(theta_vals, large_pct)
    small_adv = np.mean((snr_ctr[small_idx] - snr_fr[small_idx]) / snr_fr[small_idx])
    large_gap = np.mean(np.abs(snr_ctr[large_idx] - snr_fr[large_idx]) / snr_fr[large_idx])
    return small_adv, large_gap


if __name__ == "__main__":
    # ----- Explicit parameters (no auto-tuning) -----
    n = 256
    top_hat_width = 0.60 * np.pi  # radians, half-width of the top-hat kernel
    j_height = 1.0  # base height; J entries are j_height / N inside the top-hat
    gain_target = 0.97  # sets r*w*d*lambda_max for CTR via a global J_scale

    bump_kappa = 8.0
    rho_floor = 0.05
    rho_peak = 0.55
    bump_center = 0.0

    dtheta_min = 1e-3
    dtheta_max = np.pi
    dtheta_count = 60
    normalize_delta_x = True

    # Noise parameters chosen explicitly.
    sigma_gamma = 0.35
    alpha = 0.08

    # CTR vs FR conditions.
    r_ctr, w_ctr = 1.0, 1.0
    r_fr, w_fr = 1.27, 0.64

    # Diagnostics thresholds.
    small_pct = 25
    large_pct = 75

    # ----- Ring grid -----
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # ----- Build J: top-hat kernel with 1/N scaling, then a global scale -----
    j_base = build_top_hat_j(theta, top_hat_width, j_height)
    lam_j_base = np.linalg.eigvalsh(j_base)
    lam_max_base = lam_j_base.max()

    # Average bump response defines rho and D; since the bump shifts uniformly,
    # the spatial average is a scalar and D reduces to d * I.
    rho_profile = build_rho_profile(theta, bump_center, bump_kappa, rho_floor, rho_peak)
    rho_mean = rho_profile.mean()
    d = rho_mean * (1.0 - rho_mean)
    d_matrix = np.diag(np.full(n, d))

    # Global scaling to place the leading mode near the stability boundary.
    # To keep the raw height exactly 1/N, set gain_target = r_ctr*w_ctr*d*lam_max_base.
    j_scale = gain_target / (r_ctr * w_ctr * d * lam_max_base)
    j = j_base * j_scale

    lam_j, evecs = np.linalg.eigh(j)

    # ----- Linear response eigenvalues -----
    lam_g_ctr = gamma_eigs_from_lambda_j(lam_j, w_ctr, r_ctr, d)
    lam_g_fr = gamma_eigs_from_lambda_j(lam_j, w_fr, r_fr, d)

    # Build Gamma for CTR (used for covariance eigenvalue check).
    identity = np.eye(n)
    gamma_ctr = r_ctr * d_matrix @ np.linalg.inv(identity - r_ctr * w_ctr * d_matrix @ j)

    # Noise model:
    # Dominance condition (mode k): slow dominates fast when (w*sigma_gamma)^2 * lam_g_k >> alpha.
    # If sigma_gamma is too small and alpha is not tiny, fast noise dominates most modes -> CTR>FR everywhere.
    # Need angle-dependent mode content: if delta_x does not shift energy from low- to high-lam_g modes
    # as delta_theta increases, you cannot get "small angles differ, large angles equal" behavior.
    # Criticality vs mixing: pushing r*w*d*lambda_max toward 1 increases lam_g and favors slow dominance,
    # but corresponds to near-critical dynamics (slow mixing, huge correlations) in real networks.
    # Fast-noise scaling mismatch: the alpha*Gamma term is linear in lam_g, breaking the cancellation that
    # occurs when both signal and slow noise scale through w*Gamma.

    # ----- Figure 1: spectrum of J -----
    plt.figure(figsize=(6.0, 3.6))
    plt.plot(np.arange(n), np.sort(lam_j)[::-1], "o-", lw=1.5)
    plt.xlabel("mode index (sorted)")
    plt.ylabel("lambda_J")
    plt.title("Top-hat J spectrum")
    plt.tight_layout()

    # ----- Figure 2: lambda_J -> lambda_Gamma mapping -----
    plt.figure(figsize=(6.0, 3.6))
    plt.plot(lam_j, lam_g_ctr, "o", label="CTR mapping")
    boundary = 1.0 / (r_ctr * w_ctr * d)
    plt.axvline(boundary, color="k", ls="--", lw=1.0, label="stability boundary")
    plt.xlabel("lambda_J")
    plt.ylabel("lambda_Gamma")
    plt.title("Response gain near criticality")
    plt.legend(frameon=False)
    plt.tight_layout()

    # ----- Figure 3: covariance eigenvalues vs predicted formula -----
    sigma_ctr = (w_ctr * sigma_gamma) ** 2 * (gamma_ctr @ gamma_ctr) + alpha * gamma_ctr
    eig_sigma = np.linalg.eigvalsh(sigma_ctr)
    pred_sigma = (w_ctr * sigma_gamma) ** 2 * lam_g_ctr**2 + alpha * lam_g_ctr
    plt.figure(figsize=(6.0, 3.6))
    plt.plot(np.sort(eig_sigma), "o", label="eig(Sigma)")
    plt.plot(np.sort(pred_sigma), "-", lw=2.0, label="modewise formula")
    plt.xlabel("mode index (sorted)")
    plt.ylabel("lambda_Sigma")
    plt.title("Covariance spectrum check")
    plt.legend(frameon=False)
    plt.tight_layout()

    # ----- Figure 4: SNR vs angle difference -----
    dtheta_grid = np.linspace(dtheta_min, dtheta_max, dtheta_count)

    snr_ctr_slow = snr_curve(
        theta, evecs, lam_g_ctr, w_ctr, sigma_gamma, 0.0,
        bump_center, bump_kappa, rho_floor, rho_peak, dtheta_grid, normalize_delta_x,
    )
    snr_fr_slow = snr_curve(
        theta, evecs, lam_g_fr, w_fr, sigma_gamma, 0.0,
        bump_center, bump_kappa, rho_floor, rho_peak, dtheta_grid, normalize_delta_x,
    )
    snr_ctr_fast = snr_curve(
        theta, evecs, lam_g_ctr, w_ctr, sigma_gamma, alpha,
        bump_center, bump_kappa, rho_floor, rho_peak, dtheta_grid, normalize_delta_x,
    )
    snr_fr_fast = snr_curve(
        theta, evecs, lam_g_fr, w_fr, sigma_gamma, alpha,
        bump_center, bump_kappa, rho_floor, rho_peak, dtheta_grid, normalize_delta_x,
    )

    theta_deg = np.degrees(dtheta_grid)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    axes[0].plot(theta_deg, snr_ctr_slow, lw=2.0, label="CTR")
    axes[0].plot(theta_deg, snr_fr_slow, lw=2.0, ls="--", label="FR")
    axes[0].set_title("Slow noise only (alpha=0)")
    axes[0].set_xlabel("angle difference (deg)")
    axes[0].set_ylabel("SNR")
    axes[0].legend(frameon=False)

    axes[1].plot(theta_deg, snr_ctr_fast, lw=2.0, label="CTR")
    axes[1].plot(theta_deg, snr_fr_fast, lw=2.0, ls="--", label="FR")
    axes[1].set_title("Slow + fast noise")
    axes[1].set_xlabel("angle difference (deg)")
    axes[1].legend(frameon=False)

    fig.suptitle("SNR vs stimulus angle difference")
    plt.tight_layout()
    plt.show()

    # Diagnostics (no auto-tuning).
    slow_gap = np.mean(np.abs(snr_ctr_slow - snr_fr_slow) / snr_fr_slow)
    small_adv, large_gap = fast_noise_metrics(theta_deg, snr_ctr_fast, snr_fr_fast, small_pct, large_pct)

    print("==== Parameter summary ====")
    print(f"N={n}, top_hat_width={top_hat_width:.3f} rad, j_height={j_height}, gain_target={gain_target}")
    print(f"rho_mean={rho_mean:.3f}, d={d:.3f}, J_scale={j_scale:.3f}")
    print(f"CTR: w={w_ctr}, r={r_ctr}; FR: w={w_fr}, r={r_fr}")
    print(f"sigma_gamma={sigma_gamma}, alpha={alpha}, normalize_delta_x={normalize_delta_x}")
    print("Metrics: slow-only gap=%.3f, fast small-angle advantage=%.3f, fast large-angle gap=%.3f" %
          (slow_gap, small_adv, large_gap))
