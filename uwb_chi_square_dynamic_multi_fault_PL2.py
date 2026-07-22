"""
UWB Positioning Simulation with:
  - Time-varying random faults
  - Iterative FDE (chi-square residual test)
  - Advanced RAIM MHSS (HPL)
  - Traditional single-fault PL (Sec 2.5) - WITH UNIFIED QUANTILES
  - Section 2.7 multi-fault PL (integer optimisation) -
        * X/Y directions (original)
        * Radial (Euclidean norm) - NEW (using norm instead of e1 projection)
All previous functionality is preserved; PL methods are added for comparison.
Fixed H-robustness issue.
"""

import numpy as np
from scipy.stats import chi2, ncx2, norm
from scipy.linalg import inv, pinv
from scipy.optimize import root_scalar
import itertools
import matplotlib.pyplot as plt

# ------------------------------
# Global plot settings for single-column paper layout
# ------------------------------
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,   # Ensure editable text in PDF
})

# ------------------------------
# 1. System Configuration
# ------------------------------
# np.random.seed(42)          # change or comment for fully random runs

center = np.array([5.0, 5.0])
radius = 10.0
angles = np.deg2rad([0, 30, 60, 120, 180, 240, 300])
anchors = np.array([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])

N_anchors = anchors.shape[0]          # 6
state_dim = 2                         # x, y
sigma = 0.1                           # measurement noise std
W_full = np.diag(1.0 / sigma**2 * np.ones(N_anchors))

# Integrity / FDE parameters
alpha = 0.05          # false alarm rate for FDE
beta = 0.8            # detection probability for MDB
alpha_pl = 1e-4       # integrity risk for Advanced RAIM and ALL PL methods (UNIFIED)
AL = 1.5              # alert limit (meters)

# ------------------------------
# 2. MDB Computation
# ------------------------------
def compute_mdb(anchors, W, alpha=0.05, beta=0.8):
    N = anchors.shape[0]
    df = N - state_dim
    threshold = chi2.ppf(1 - alpha, df)

    pos0 = np.array([5.0, 5.0])
    H = np.zeros((N, state_dim))
    for i in range(N):
        diff = pos0 - anchors[i]
        norm_val = np.linalg.norm(diff)
        if norm_val > 1e-6:
            H[i, :] = diff / norm_val

    HtWH = H.T @ W @ H
    try:
        HtWH_inv = inv(HtWH)
    except np.linalg.LinAlgError:
        HtWH_inv = pinv(HtWH)
    S = W - W @ H @ HtWH_inv @ H.T @ W
    s_diag = np.diag(S)
    min_s = 1e-6
    s_diag = np.maximum(s_diag, min_s)

    def func(lam):
        return ncx2.sf(threshold, df, lam) - beta

    lam_low = 0.0
    lam_high = 50.0
    while ncx2.sf(threshold, df, lam_high) > beta:
        lam_high *= 2
    sol = root_scalar(func, bracket=[lam_low, lam_high], method='bisect')
    lambda_req = sol.root

    mdb = np.sqrt(lambda_req / s_diag)
    return mdb, s_diag, lambda_req

mdb_values, s_diag, lambda_req = compute_mdb(anchors, W_full, alpha, beta)

# ------------------------------
# 3. Core Positioning Functions
# ------------------------------
def compute_ranges(pos, anchors):
    return np.sqrt(np.sum((anchors - pos)**2, axis=1))

def wls_position(measurements, anchors, W):
    """
    Weighted Least-Squares via Gauss-Newton.
    Returns (position, Jacobian H). H is always a 2D array (m x 2) if m>=2.
    """
    m = len(anchors)
    if m < 2:
        return np.mean(anchors, axis=0), None
    x = np.mean(anchors, axis=0)
    H = None  # ensure defined even if loop fails
    for _ in range(10):
        r = compute_ranges(x, anchors)
        residuals = measurements - r
        H = np.zeros((m, 2))
        for i in range(m):
            diff = x - anchors[i]
            norm_val = np.linalg.norm(diff)
            if norm_val > 1e-6:
                H[i, :] = diff / norm_val
        HtWH = H.T @ W @ H
        HtWres = H.T @ W @ residuals
        try:
            delta = inv(HtWH) @ HtWres
        except np.linalg.LinAlgError:
            break
        x += delta
        if np.linalg.norm(delta) < 1e-6:
            break
    return x, H

def compute_wsse(measurements, pos_est, anchors, W):
    r_est = compute_ranges(pos_est, anchors)
    e = measurements - r_est
    return e.T @ W @ e

def compute_standardised_residuals(measurements, pos_est, anchors, W):
    m = len(anchors)
    r_est = compute_ranges(pos_est, anchors)
    e = measurements - r_est
    H = np.zeros((m, 2))
    for i in range(m):
        diff = pos_est - anchors[i]
        norm_val = np.linalg.norm(diff)
        if norm_val > 1e-6:
            H[i, :] = diff / norm_val
    try:
        P = inv(H.T @ W @ H)
        Q_res = inv(W) - H @ P @ H.T
        std_res = np.sqrt(np.diag(Q_res))
    except np.linalg.LinAlgError:
        std_res = sigma * np.ones(m)
    std_res[std_res < 1e-9] = sigma
    return np.abs(e) / std_res

# ------------------------------
# 4. Iterative FDE
# ------------------------------
def detect_and_exclude(measurements, anchors, W, alpha, min_anchors = 4):
    n = len(anchors)
    keep_indices = list(range(n))
    excluded = False

    while len(keep_indices) >= min_anchors:
        cur_anchors = anchors[keep_indices]
        cur_meas = measurements[keep_indices]
        cur_W = np.diag(1.0 / sigma**2 * np.ones(len(keep_indices)))

        pos_est, H = wls_position(cur_meas, cur_anchors, cur_W)
        if H is None or not hasattr(H, 'shape') or H.ndim < 2 or H.shape[0] < 2:
            break

        r = cur_meas - compute_ranges(pos_est, cur_anchors)
        SSE = r @ cur_W @ r
        dof = len(keep_indices) - 2
        if dof <= 0:
            break
        threshold = chi2.ppf(1 - alpha, dof)

        if SSE < threshold:
            return keep_indices, excluded

        try:
            P = inv(H.T @ cur_W @ H)
        except np.linalg.LinAlgError:
            P = pinv(H.T @ cur_W @ H)
        R = inv(cur_W) - H @ P @ H.T
        diag_R = np.diag(R)
        std_resid = np.abs(r) / np.sqrt(np.maximum(diag_R, 1e-12))
        idx_local = np.argmax(std_resid)
        idx_global = keep_indices[idx_local]
        keep_indices.pop(idx_local)
        excluded = True

    if len(keep_indices) < min_anchors:
        return list(range(n)), excluded
    return keep_indices, excluded

# ------------------------------
# 5. Advanced RAIM MHSS
# ------------------------------
def compute_advanced_raim_pl(measurements, anchors, W, alpha_pl, AL):
    """
    Advanced RAIM (ARAIM) MHSS HPL computation.
    *** UPDATED: Operates on the currently used (kept) subset only. ***
    Assumes measurements/anchors are already the FDE-cleaned set.
    """
    n = len(anchors)  # This is the number of KEPT measurements

    # If fewer than 3 anchors, ARAIM cannot provide meaningful protection
    if n < 3:
        return 0.0, False, []

    # 1. Full-set solution (which is the current pos_est)
    full_pos = wls_position(measurements, anchors, W)[0]

    # 2. Compute Jacobian for the kept subset
    H_full = np.zeros((n, 2))
    for i in range(n):
        diff = full_pos - anchors[i]
        norm_val = np.linalg.norm(diff)
        if norm_val > 1e-6:
            H_full[i, :] = diff / norm_val

    try:
        P_full = inv(H_full.T @ W @ H_full)
    except np.linalg.LinAlgError:
        P_full = np.eye(2) * 1e6

    subsets = []
    subsets.append(('full', list(range(n)), full_pos, P_full))

    # 3. Single-fault exclusion subsets (only among the kept anchors)
    for excl in range(n):
        keep = [i for i in range(n) if i != excl]
        if len(keep) < state_dim + 1:
            continue

        meas_sub = measurements[keep]
        anch_sub = anchors[keep]
        W_sub = W[keep][:, keep]
        pos_sub = wls_position(meas_sub, anch_sub, W_sub)[0]

        H_sub = np.zeros((len(keep), 2))
        for i, idx in enumerate(keep):
            diff = pos_sub - anchors[idx]
            norm_val = np.linalg.norm(diff)
            if norm_val > 1e-6:
                H_sub[i, :] = diff / norm_val
        try:
            P_sub = inv(H_sub.T @ W_sub @ H_sub)
        except np.linalg.LinAlgError:
            P_sub = np.eye(2) * 1e6

        subsets.append((f'excl_{excl}', keep, pos_sub, P_sub))

    # 4. Bonferroni correction for the number of hypotheses in THIS subset
    M = len(subsets)  # Number of hypotheses is n+1, not N+1
    k_md = norm.ppf(1 - alpha_pl / M)

    hpl_list = []
    for _, keep, pos_sub, P_sub in subsets:
        eigenvals = np.linalg.eigvalsh(P_sub)
        sigma_h = np.sqrt(np.max(eigenvals))
        bias = np.linalg.norm(pos_sub - full_pos)
        hpl_i = k_md * sigma_h + bias
        hpl_list.append(hpl_i)

    hpl = max(hpl_list)
    alarm = hpl > AL
    return hpl, alarm, subsets

# ------------------------------
# 6. Protection Level Calculators (UNIFIED QUANTILES)
# ------------------------------
def compute_pl_traditional(H, W, chi2_threshold, alpha_pl):
    """
    Traditional single-fault PL (Section 2.5).
    Noise quantile is unified using Bonferroni correction over N single faults.
    """
    n = H.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Unified quantile: split risk among N single-fault hypotheses
    k_noise = norm.ppf(1 - alpha_pl / n)

    try:
        P = inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        P = pinv(H.T @ W @ H)
    PL_noise_x = k_noise * np.sqrt(P[0, 0])
    PL_noise_y = k_noise * np.sqrt(P[1, 1])

    try:
        HtWH_inv = inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        HtWH_inv = pinv(H.T @ W @ H)
    S = W - W @ H @ HtWH_inv @ H.T @ W

    Cx = np.array([[1, 0], [0, 0]])
    Cy = np.array([[0, 0], [0, 1]])
    WH = W @ H
    HtWH_inv_Ht = HtWH_inv @ H.T

    PL_fault_x = 0.0
    PL_fault_y = 0.0

    for direction, C in enumerate([Cx, Cy]):
        D = WH @ HtWH_inv @ C @ HtWH_inv_Ht @ W
        for j in range(n):
            gamma = np.zeros(n)
            gamma[j] = 1.0
            denom = gamma @ S @ gamma
            if denom < 1e-12:
                continue
            num = gamma @ D @ gamma
            if num < 0:
                continue
            bias = np.sqrt(chi2_threshold * num / denom)
            if direction == 0:
                if bias > PL_fault_x:
                    PL_fault_x = bias
            else:
                if bias > PL_fault_y:
                    PL_fault_y = bias

    PL_x = PL_noise_x + PL_fault_x
    PL_y = PL_noise_y + PL_fault_y
    return PL_x, PL_y, PL_noise_x, PL_noise_y, PL_fault_x, PL_fault_y

def compute_pl_section27_with_details(H, W, chi2_threshold, max_faults, alpha_pl):
    """
    Section 2.7 multi-fault PL (integer optimisation) - X/Y directions.
    Noise quantile is unified using Bonferroni correction over ALL fault combinations
    (r=1 to max_faults).
    """
    n = H.shape[0]
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                None, None, None, None)

    # Count total number of fault hypotheses for Bonferroni correction
    total_hyp = 0
    for r in range(1, max_faults + 1):
        if r <= n:
            total_hyp += len(list(itertools.combinations(range(n), r)))
    if total_hyp == 0:
        total_hyp = 1  # fallback

    # UNIFIED quantile: split risk among all multi-fault combinations
    k_noise = norm.ppf(1 - alpha_pl / total_hyp)

    try:
        P = inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        P = pinv(H.T @ W @ H)
    PL_noise_x = k_noise * np.sqrt(P[0, 0])
    PL_noise_y = k_noise * np.sqrt(P[1, 1])

    try:
        HtWH_inv = inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        HtWH_inv = pinv(H.T @ W @ H)
    S = W - W @ H @ HtWH_inv @ H.T @ W
    Mx = (HtWH_inv @ H.T @ W)[0, :]
    My = (HtWH_inv @ H.T @ W)[1, :]

    best_fault_x = 0.0
    best_fault_y = 0.0
    best_d1_x = None
    best_d2_x = None
    best_d1_y = None
    best_d2_y = None

    for r in range(1, max_faults + 1):
        if r > n:
            break
        for combo in itertools.combinations(range(n), r):
            S_sub = S[np.ix_(combo, combo)]
            sum_abs_S = np.sum(np.abs(S_sub))
            if sum_abs_S < 1e-12:
                continue

            D_val = np.sqrt(chi2_threshold / sum_abs_S)

            # X direction
            sum_abs_Mx = np.sum([np.abs(Mx[i]) for i in combo])
            bias_x = 2 * D_val * sum_abs_Mx / 2
            if bias_x > best_fault_x:
                best_fault_x = bias_x
                d1 = np.zeros(n)
                d2 = np.zeros(n)
                for idx in combo:
                    sign = np.sign(Mx[idx]) if abs(Mx[idx]) > 1e-12 else 0.0
                    d1[idx] = -D_val * sign
                    d2[idx] = +D_val * sign
                best_d1_x = d1
                best_d2_x = d2

            # Y direction
            sum_abs_My = np.sum([np.abs(My[i]) for i in combo])
            bias_y = 2 * D_val * sum_abs_My / 2
            if bias_y > best_fault_y:
                best_fault_y = bias_y
                d1 = np.zeros(n)
                d2 = np.zeros(n)
                for idx in combo:
                    sign = np.sign(My[idx]) if abs(My[idx]) > 1e-12 else 0.0
                    d1[idx] = -D_val * sign
                    d2[idx] = +D_val * sign
                best_d1_y = d1
                best_d2_y = d2

    PL_x = PL_noise_x + best_fault_x
    PL_y = PL_noise_y + best_fault_y
    return (PL_x, PL_y, PL_noise_x, PL_noise_y, best_fault_x, best_fault_y,
            best_d1_x, best_d2_x, best_d1_y, best_d2_y)

# ------------------------------------------------------------
# NEW: Section 2.7 PL using Euclidean norm (Radial) - norm instead of e1
# ------------------------------------------------------------
def compute_pl_section27_radial(H, W, chi2_threshold, max_faults, alpha_pl):
    """
    Section 2.7 multi-fault PL for Euclidean norm (Radial).
    Returns:
        PL_radial  : float, total radial protection level
        best_bias  : float, max radial bias magnitude
    """
    n = H.shape[0]
    if n == 0:
        return 0.0, 0.0

    # 1. Count total fault hypotheses for Bonferroni correction
    total_hyp = 0
    for r in range(1, max_faults + 1):
        if r <= n:
            total_hyp += len(list(itertools.combinations(range(n), r)))
    if total_hyp == 0:
        total_hyp = 1

    k_noise = norm.ppf(1 - alpha_pl / total_hyp)

    # 2. Noise-induced radial PL (using trace of covariance)
    try:
        P = inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        P = pinv(H.T @ W @ H)
    PL_noise_radial = k_noise * np.sqrt(np.trace(P))

    # 3. Fault-induced radial PL
    try:
        HtWH_inv = inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        HtWH_inv = pinv(H.T @ W @ H)

    M = HtWH_inv @ H.T @ W          # shape (2, n)
    S = W - W @ H @ HtWH_inv @ H.T @ W

    best_bias = 0.0

    for r in range(1, max_faults + 1):
        if r > n:
            break
        for combo in itertools.combinations(range(n), r):
            S_sub = S[np.ix_(combo, combo)]
            sum_abs_S = np.sum(np.abs(S_sub))
            if sum_abs_S < 1e-12:
                continue

            D_val = np.sqrt(chi2_threshold / sum_abs_S)

            # Sum of L2 norms of columns of M for this combo
            sum_norm_M = 0.0
            for idx in combo:
                sum_norm_M += np.linalg.norm(M[:, idx])

            bias_radial = D_val * sum_norm_M
            if bias_radial > best_bias:
                best_bias = bias_radial

    PL_radial = PL_noise_radial + best_bias
    return PL_radial, best_bias

# ------------------------------
# 7. Helper: generate time-varying faults
# ------------------------------
def generate_bias_sequence(num_steps, fault_start, fault_end,
                           amplitude, frequency, phase):
    t = np.arange(num_steps)
    duration = fault_end - fault_start
    if duration <= 0:
        return np.zeros(num_steps)
    seq = amplitude * np.sin(2 * np.pi * frequency * (t - fault_start) / duration + phase)
    seq[:fault_start] = 0.0
    seq[fault_end:] = 0.0
    return seq

# ------------------------------
# 8. Main Simulation
# ------------------------------
def run_simulation():
    num_steps = 1000
    t = np.linspace(0, 10 * np.pi, num_steps)
    true_traj = np.column_stack((5 + 6 * np.cos(t), 5 + 4 * np.sin(2 * t)))

    # ---- Random time-varying fault configuration ----
    min_faults = 2
    max_faults = 2
    fault_start, fault_end = 150, 800

    num_faults = np.random.randint(min_faults, max_faults + 1)
    fault_indices = np.random.choice(N_anchors, size=num_faults, replace=False)
    fault_bias_sequences = {}
    fault_params = {}
    for idx in fault_indices:
        amplitude = np.random.uniform(0.5, 1.6) * np.random.choice([-1, 1])
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        seq = generate_bias_sequence(num_steps, fault_start, fault_end,
                                     amplitude, frequency, phase)
        fault_bias_sequences[idx] = seq
        fault_params[idx] = (amplitude, frequency, phase)

    print("\n=== Random Time-Varying Faults ===")
    for idx in fault_indices:
        amp, freq, ph = fault_params[idx]
        print(f"Anchor {idx}: amplitude={amp:+.2f}m, frequency={freq:.2f} cycles/duration")

    max_abs_bias = {idx: np.max(np.abs(seq)) for idx, seq in fault_bias_sequences.items()}
    print("\n=== MDB & Fault Detectability ===")
    for idx in fault_indices:
        print(f"Anchor {idx}: max|bias|={max_abs_bias[idx]:.3f}m, MDB={mdb_values[idx]:.4f}m -> "
              f"{'Detectable' if max_abs_bias[idx] > mdb_values[idx] else 'Not detectable'}")

    # ---- Storage arrays ----
    est_traj = np.zeros((num_steps, 2))
    true_error = np.zeros((num_steps, 2))
    hpl_vals = np.zeros(num_steps)          # Advanced RAIM HPL
    alarm_advanced = np.zeros(num_steps, dtype=bool)
    PL_x_trad = np.zeros(num_steps)
    PL_y_trad = np.zeros(num_steps)
    PL_x_new = np.zeros(num_steps)
    PL_y_new = np.zeros(num_steps)
    PL_radial_new = np.zeros(num_steps)     # NEW: Radial PL from Section 2.7
    fault_active = np.zeros(num_steps, dtype=bool)
    excluded_history = []

    # Storage for optimal d vectors (optional)
    d1_x_hist = np.full((num_steps, N_anchors), np.nan)
    d2_x_hist = np.full((num_steps, N_anchors), np.nan)
    d1_rad_hist = np.full((num_steps, N_anchors), np.nan)
    d2_rad_hist = np.full((num_steps, N_anchors), np.nan)

    # ---- Simulation loop ----
    for k in range(num_steps):
        true_pos = true_traj[k]
        true_ranges = compute_ranges(true_pos, anchors)
        noise = np.random.normal(0, sigma, N_anchors)
        measurements = true_ranges + noise

        if fault_start <= k < fault_end:
            for idx in fault_indices:
                measurements[idx] += fault_bias_sequences[idx][k]
            fault_active[k] = True

        # FDE step
        keep_idx, _ = detect_and_exclude(measurements, anchors, W_full, alpha, min_anchors=4)
        kept_anchors = anchors[keep_idx]
        kept_meas = measurements[keep_idx]
        n_kept = len(keep_idx)

        if n_kept < 2:
            kept_anchors = anchors
            kept_meas = measurements
            n_kept = N_anchors

        W_kept = np.diag(1.0 / sigma**2 * np.ones(n_kept))

        # WLS estimate and Jacobian
        pos_est, H = wls_position(kept_meas, kept_anchors, W_kept)

        # Robustness: if H is invalid, build it manually
        if H is None or not hasattr(H, 'shape') or H.ndim < 2 or H.shape[0] < 2:
            pos_est = np.mean(kept_anchors, axis=0)
            H = np.zeros((n_kept, 2))
            for i in range(n_kept):
                diff = pos_est - kept_anchors[i]
                norm_val = np.linalg.norm(diff)
                if norm_val > 1e-6:
                    H[i, :] = diff / norm_val

        est_traj[k] = pos_est
        true_error[k] = pos_est - true_pos
        excluded_history.append([i for i in range(N_anchors) if i not in keep_idx])

        # Chi-square threshold for the kept subset
        dof_kept = n_kept - 2
        chi2_thr = chi2.ppf(1 - alpha, dof_kept) if dof_kept > 0 else 0.0

        # Advanced RAIM HPL based on the KEPT subset
        hpl, alarm, _ = compute_advanced_raim_pl(kept_meas, kept_anchors, W_kept, alpha_pl, AL)
        hpl_vals[k] = hpl
        alarm_advanced[k] = alarm

        # Traditional PL (single fault) - WITH UNIFIED QUANTILE
        PL_x_trad[k], PL_y_trad[k], _, _, _, _ = compute_pl_traditional(
            H, W_kept, chi2_thr, alpha_pl
        )

        # Section 2.7 PL (multi-fault) - X/Y directions
        (PL_x_new[k], PL_y_new[k], _, _, _, _,
         d1_x, d2_x, d1_y, d2_y) = compute_pl_section27_with_details(
            H, W_kept, chi2_thr, max_faults=2, alpha_pl=alpha_pl
        )

        # ---- NEW: Section 2.7 Radial PL (Euclidean norm) ----
        PL_radial_new[k], bias_rad = compute_pl_section27_radial(
            H, W_kept, chi2_thr, max_faults=2, alpha_pl=alpha_pl
        )

    # ---- Post-processing and plots ----
    time = np.arange(num_steps)
    error_norm = np.linalg.norm(true_error, axis=1)

    # Single-column width: 3.5 inches
    fig_width = 3.5
    fig_height = 2.5

    # --------------------------------------------------
    # Figure 1: Trajectory overview
    # --------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6.8, 5.5))
    ax1.plot(true_traj[:, 0], true_traj[:, 1], 'k-', lw=1.5, label='True')
    ax1.plot(est_traj[:, 0], est_traj[:, 1], 'b--', lw=1.0, label='Estimated (with FDE)')
    fault_idx = np.where(fault_active)[0]
    if len(fault_idx) > 0:
        ax1.plot(true_traj[fault_idx, 0], true_traj[fault_idx, 1],
                 'r-', lw=3, alpha=0.2, label='Fault active')
    ax1.scatter(anchors[:, 0], anchors[:, 1], c='green', s=30, marker='^', label='Anchors')
    for idx in fault_indices:
        amp, freq, ph = fault_params[idx]
        ax1.annotate(f'F{idx}\n({amp:+.1f}m)',
                     xy=anchors[idx], xytext=(5,5), textcoords='offset points',
                     fontsize=6, color='darkred')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory with Time-Varying Faults and FDE')
    ax1.grid(True); ax1.axis('equal'); ax1.legend(fontsize=6, loc='best')
    fig1.tight_layout()
    fig1.savefig('fig_trajectory.pdf', format='pdf')
    plt.close(fig1)

    # --------------------------------------------------
    # Figure 2: Error vs PL in X and Y directions (two stacked subplots)
    # --------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(6.8, 4.5), sharex=True)
    # Subplot 1: X-direction
    ax2a.plot(time, abs(true_error[:, 0]), 'b-', lw=0.8, label='|Error X|')
    # ax2a.plot(time, PL_x_trad, 'r-', lw=1.0, label='Trad PL X')
    ax2a.plot(time, PL_x_new, 'g-', lw=1.0, label='PL X')
    ax2a.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax2a.set_ylabel('Error / PL (m)')
    ax2a.legend(fontsize=6, loc='upper left')
    ax2a.grid(True)
    ax2a.set_title('X-direction', fontsize=8)
    # Subplot 2: Y-direction
    ax2b.plot(time, abs(true_error[:, 1]), 'b-', lw=0.8, label='|Error Y|')
    # ax2b.plot(time, PL_y_trad, 'r-', lw=1.0, label='Trad PL Y')
    ax2b.plot(time, PL_y_new, 'g-', lw=1.0, label='PL Y')
    ax2b.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax2b.set_xlabel('Time step')
    ax2b.set_ylabel('Error / PL (m)')
    ax2b.legend(fontsize=6, loc='upper left')
    ax2b.grid(True)
    ax2b.set_title('Y-direction', fontsize=8)
    fig2.tight_layout()
    fig2.savefig('fig_xy_pl.pdf', format='pdf')
    plt.close(fig2)

    # --------------------------------------------------
    # Figure 3: Radial error vs various PLs (single plot)
    # --------------------------------------------------
    PL_norm_trad = np.sqrt(PL_x_trad**2 + PL_y_trad**2)
    PL_norm_new_xy = np.sqrt(PL_x_new**2 + PL_y_new**2)

    fig3, ax3 = plt.subplots(figsize=(6.8, 3.5))
    # ax3.plot(time, PL_norm_trad, 'r--', lw=0.8, label='Trad |PL|')
    # ax3.plot(time, PL_norm_new_xy, 'c--', lw=0.8, label='Sec 2.7 |PL| (X/Y)')
    ax3.plot(time, PL_radial_new, 'g-', lw=1.2, label='Radial PL')
    ax3.plot(time, hpl_vals, 'm-', lw=0.8, label='ARAIM HPL')
    ax3.plot(time, error_norm, 'b-', lw=0.8, label='|Error| (Euclidean)')
    ax3.axhline(y=AL, color='k', linestyle=':', lw=0.8, label=f'AL = {AL} m')
    ax3.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Radial error / PL (m)')
    ax3.legend(fontsize=6, loc='upper left', ncol=2)
    ax3.grid(True)
    ax3.set_title('Radial Error vs. Protection Levels', fontsize=8)
    fig3.tight_layout()
    fig3.savefig('fig_radial_pl.pdf', format='pdf')
    plt.close(fig3)

    # ---- Statistics ----
    fault_steps = np.sum(fault_active)
    print("\n=== Performance Summary ===")
    print(f"Faulty anchors: {fault_indices}")
    print(f"Fault duration: steps {fault_start} to {fault_end-1}")
    print(f"Steps with faults: {fault_steps}")

    # Advanced RAIM
    total_araim_alarms = np.sum(alarm_advanced)
    false_araim = np.sum(alarm_advanced & ~fault_active)
    missed_araim = np.sum(~alarm_advanced & fault_active)
    print(f"\nAdvanced RAIM (HPL > {AL} m):")
    print(f"  Alarms raised: {total_araim_alarms}")
    print(f"  False alarms: {false_araim}")
    print(f"  Missed detections: {missed_araim}")
    if fault_steps > 0:
        print(f"  Detection rate: {1 - missed_araim/fault_steps:.3f}")
    print(f"  Mean HPL: {np.mean(hpl_vals):.3f} m, Max HPL: {np.max(hpl_vals):.3f} m")

    # FDE
    excluded_counts = [len(e) for e in excluded_history]
    print(f"\nFDE exclusions: mean {np.mean(excluded_counts):.2f}, max {np.max(excluded_counts)}")
    fault_excl = [excl for k, excl in enumerate(excluded_history) if fault_active[k]]
    if fault_excl:
        print(f"Avg excluded anchors during faults: {np.mean([len(e) for e in fault_excl]):.2f}")

    # PL coverage
    cover_trad_x = np.all(np.abs(true_error[:, 0]) <= PL_x_trad)
    cover_trad_y = np.all(np.abs(true_error[:, 1]) <= PL_y_trad)
    cover_new_x = np.all(np.abs(true_error[:, 0]) <= PL_x_new)
    cover_new_y = np.all(np.abs(true_error[:, 1]) <= PL_y_new)
    cover_norm_trad = np.all(error_norm <= PL_norm_trad)
    cover_norm_new_xy = np.all(error_norm <= PL_norm_new_xy)
    cover_radial_new = np.all(error_norm <= PL_radial_new)

    print("\n=== PL Coverage Check (UNIFIED Quantiles) ===")
    print(f"Traditional PL (single fault):")
    print(f"  X: {cover_trad_x}, Y: {cover_trad_y}, Norm (from X/Y): {cover_norm_trad}")
    print(f"Section 2.7 PL (integer opt, X/Y):")
    print(f"  X: {cover_new_x}, Y: {cover_new_y}, Norm (from X/Y): {cover_norm_new_xy}")
    print(f"Section 2.7 Radial PL (Euclidean norm):")
    print(f"  Radial: {cover_radial_new}")

    print(f"\nMean absolute error: X = {np.mean(np.abs(true_error[:,0])):.3f} m, "
          f"Y = {np.mean(np.abs(true_error[:,1])):.3f} m")
    print(f"Mean PL (traditional): X = {np.mean(PL_x_trad):.3f} m, Y = {np.mean(PL_y_trad):.3f} m")
    print(f"Mean PL (Section 2.7 X/Y): X = {np.mean(PL_x_new):.3f} m, Y = {np.mean(PL_y_new):.3f} m")
    print(f"Mean PL (Section 2.7 Radial): {np.mean(PL_radial_new):.3f} m")

    # ---- Additional stats for Radial PL and ARAIM (Euclidean norm) ----
    cover_araim_radial = np.mean(error_norm <= hpl_vals) * 100
    cover_radial_new = np.mean(error_norm <= PL_radial_new) * 100
    mean_hpl = np.mean(hpl_vals)
    max_hpl = np.max(hpl_vals)
    mean_pl_radial = np.mean(PL_radial_new)
    max_pl_radial = np.max(PL_radial_new)

    print("\n=== Radial PL vs ARAIM HPL (Euclidean norm) ===")
    print(f"ARAIM HPL: Coverage = {cover_araim_radial:.1f}%, Mean = {mean_hpl:.3f} m, Max = {max_hpl:.3f} m")
    print(f"Section 2.7 Radial PL: Coverage = {cover_radial_new:.1f}%, Mean = {mean_pl_radial:.3f} m, Max = {max_pl_radial:.3f} m")

if __name__ == "__main__":
    run_simulation()