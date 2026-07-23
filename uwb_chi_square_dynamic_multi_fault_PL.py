"""
UWB Positioning Simulation with:
  - Time‑varying random faults
  - Iterative FDE (chi‑square residual test)
  - Advanced RAIM MHSS (HPL)
  - Traditional single‑fault PL (Sec 2.5) - WITH UNIFIED QUANTILES
  - Corrected Section 2.7 multi‑fault PL (integer optimisation) - WITH UNIFIED QUANTILES
All previous functionality is preserved; PL methods are added for comparison.
Fixed H‑robustness issue.
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
# np.random.seed(42)          

center = np.array([5.0, 5.0])
radius = 10.0
angles = np.deg2rad([0, 30, 60, 120, 180, 240, 300])
anchors = np.array([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])

N_anchors = anchors.shape[0] # 6
state_dim = 2                         # x, y
sigma = 0.1                            # measurement noise std
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
    Weighted Least‑Squares via Gauss‑Newton.
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
    Section 2.7 multi-fault PL (integer optimisation).
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

# ------------------------------
# 7. Helper: generate time‑varying faults
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

    # ---- Random time‑varying fault configuration ----
    min_faults = 2
    max_faults = 2
    fault_start, fault_end = 150, 800

    num_faults = np.random.randint(min_faults, max_faults + 1)
    fault_indices = np.random.choice(N_anchors, size=num_faults, replace=False)
    fault_bias_sequences = {}
    fault_params = {}
    for idx in fault_indices:
        amplitude = np.random.uniform(0.5, 2.9) * np.random.choice([-1, 1])
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        seq = generate_bias_sequence(num_steps, fault_start, fault_end,
                                     amplitude, frequency, phase)
        fault_bias_sequences[idx] = seq
        fault_params[idx] = (amplitude, frequency, phase)

    print("\n=== Random Time‑Varying Faults ===")
    for idx in fault_indices:
        amp, freq, ph = fault_params[idx]
        print(f"Anchor {idx}: amplitude={amp:+.2f}m, frequency={freq:.2f} cycles/duration")

    max_abs_bias = {idx: np.max(np.abs(seq)) for idx, seq in fault_bias_sequences.items()}
    print("\n=== MDB & Fault Detectability ===")
    for idx in fault_indices:
        print(f"Anchor {idx}: max|bias|={max_abs_bias[idx]:.3f}m, MDB={mdb_values[idx]:.4f}m -> "
              f"{'Detectable' if max_abs_bias[idx] > mdb_values[idx] else 'Not detectable'}")

    # ---- Storage ----
    est_traj = np.zeros((num_steps, 2))
    true_error = np.zeros((num_steps, 2))
    hpl_vals = np.zeros(num_steps)          # Advanced RAIM HPL
    alarm_advanced = np.zeros(num_steps, dtype=bool)
    PL_x_trad = np.zeros(num_steps)
    PL_y_trad = np.zeros(num_steps)
    PL_x_new = np.zeros(num_steps)
    PL_y_new = np.zeros(num_steps)
    fault_active = np.zeros(num_steps, dtype=bool)
    excluded_history = []

    # Storage for corrected state estimates from optimal d1,d2
    x1_x_hist = np.full((num_steps, 2), np.nan)
    x2_x_hist = np.full((num_steps, 2), np.nan)
    x1_y_hist = np.full((num_steps, 2), np.nan)
    x2_y_hist = np.full((num_steps, 2), np.nan)
    # Storage for d1_x, d2_x vectors
    d1_x_hist = np.full((num_steps, N_anchors), np.nan)
    d2_x_hist = np.full((num_steps, N_anchors), np.nan)

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

        # Chi‑square threshold for the kept subset
        dof_kept = n_kept - 2
        chi2_thr = chi2.ppf(1 - alpha, dof_kept) if dof_kept > 0 else 0.0

       # Advanced RAIM HPL based on the KEPT subset (consistent with pos_est)
        # Only protect against single faults among the remaining healthy anchors
        hpl, alarm, _ = compute_advanced_raim_pl(kept_meas, kept_anchors, W_kept, alpha_pl, AL)

        hpl_vals[k] = hpl
        alarm_advanced[k] = alarm

        # Traditional PL (single fault) - WITH UNIFIED QUANTILE
        PL_x_trad[k], PL_y_trad[k], _, _, _, _ = compute_pl_traditional(
            H, W_kept, chi2_thr, alpha_pl
        )

        # Section 2.7 PL (multi‑fault) - WITH UNIFIED QUANTILE
        (PL_x_new[k], PL_y_new[k], _, _, _, _,
         d1_x, d2_x, d1_y, d2_y) = compute_pl_section27_with_details(
            H, W_kept, chi2_thr, max_faults=2, alpha_pl=alpha_pl
        )

        # ---- Compute the corresponding state estimates using correct linearisation ----
        delta_r = kept_meas - compute_ranges(pos_est, kept_anchors)
        try:
            HtWH_inv = inv(H.T @ W_kept @ H)
        except np.linalg.LinAlgError:
            HtWH_inv = pinv(H.T @ W_kept @ H)
        G = HtWH_inv @ H.T @ W_kept

        if d1_x is not None:
            dx1 = G @ (delta_r - d1_x)
            dx2 = G @ (delta_r - d2_x)
            x1_x_hist[k] = pos_est + dx1
            x2_x_hist[k] = pos_est + dx2
            d1_full = np.zeros(N_anchors)
            d2_full = np.zeros(N_anchors)
            for i, idx in enumerate(keep_idx):
                d1_full[idx] = d1_x[i]
                d2_full[idx] = d2_x[i]
            d1_x_hist[k] = d1_full
            d2_x_hist[k] = d2_full

        if d1_y is not None:
            dy1 = G @ (delta_r - d1_y)
            dy2 = G @ (delta_r - d2_y)
            x1_y_hist[k] = pos_est + dy1
            x2_y_hist[k] = pos_est + dy2

    # ---- Post‑processing and plots ----
    time = np.arange(num_steps)
    error_norm = np.linalg.norm(true_error, axis=1)

    # Plot 1: Trajectory overview
    plt.figure(figsize=(12, 8))
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'k-', lw=2, label='True')
    plt.plot(est_traj[:, 0], est_traj[:, 1], 'b--', lw=1.5, label='Estimated (with FDE)')
    fault_idx = np.where(fault_active)[0]
    if len(fault_idx) > 0:
        plt.plot(true_traj[fault_idx, 0], true_traj[fault_idx, 1],
                 'r-', lw=4, alpha=0.2, label='Fault active')
    plt.scatter(anchors[:, 0], anchors[:, 1], c='green', s=120, marker='^', label='Anchors')
    for idx in fault_indices:
        amp, freq, ph = fault_params[idx]
        plt.annotate(f'Faulty A{idx}\n(amp={amp:+.1f}m)',
                     xy=anchors[idx], xytext=(10,10), textcoords='offset points',
                     fontsize=9, color='darkred')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.title('UWB Trajectory with Time‑Varying Faults and FDE')
    plt.grid(True); plt.axis('equal'); plt.legend()
    plt.tight_layout()

    # Plot 2: Error vs PL comparison (X and Y)
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(time, abs(true_error[:, 0]), 'b-', lw=1.5, label='Error X')
    # plt.plot(time, PL_x_trad, 'r-', lw=2, label='Trad PL X (unified)')
    plt.plot(time, PL_x_new, 'g-', lw=2, label='Sec 2.7 PL X (unified)')
    # plt.plot(time, hpl_vals, 'm-', lw=1.5, label='Advanced RAIM HPL')
    plt.axvspan(fault_start, fault_end-1, alpha=0.15, color='red', label='Faults')
    plt.ylabel('Error / PL (m)'); plt.legend(); plt.grid(True)
    plt.title('X‑direction: Error vs. Protection Levels (ALL with UNIFIED Quantiles)')

    plt.subplot(2, 1, 2)
    plt.plot(time, abs(true_error[:, 1]), 'b-', lw=1.5, label='Error Y')
    # plt.plot(time, PL_y_trad, 'r-', lw=2, label='Trad PL Y (unified)')
    plt.plot(time, PL_y_new, 'g-', lw=2, label='Sec 2.7 PL Y (unified)')
    # plt.plot(time, hpl_vals, 'm-', lw=1.5, label='Advanced RAIM HPL')
    plt.axvspan(fault_start, fault_end-1, alpha=0.15, color='red', label='Faults')
    plt.xlabel('Time step'); plt.ylabel('Error / PL (m)'); plt.legend(); plt.grid(True)
    plt.title('Y‑direction: Error vs. Protection Levels (ALL with UNIFIED Quantiles)')
    plt.tight_layout()

    # Plot 3: Euclidean norm comparison
    PL_norm_trad = np.sqrt(PL_x_trad**2 + PL_y_trad**2)
    PL_norm_new = np.sqrt(PL_x_new**2 + PL_y_new**2)

    plt.figure(figsize=(12, 5))
    plt.plot(time, error_norm, 'b-', lw=1.5, label='|Error|')
    # plt.plot(time, PL_norm_trad, 'r-', lw=2, label='Trad |PL| (unified)')
    plt.plot(time, PL_norm_new, 'g-', lw=2, label='Sec 2.7 |PL| (unified)')
    plt.plot(time, hpl_vals, 'm-', lw=1.5, label='Advanced RAIM HPL')
    plt.axhline(y=AL, color='k', linestyle=':', label=f'Alert Limit = {AL} m')
    plt.axvspan(fault_start, fault_end-1, alpha=0.15, color='red', label='Faults')
    plt.xlabel('Time step'); plt.ylabel('Euclidean norm (m)')
    plt.title('Euclidean Error vs. Protection Level Norms (UNIFIED Quantiles)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()

    # ---- Plot d1_x and d2_x over time ----
    plt.figure(figsize=(6.8, 3.5))

    plt.subplot(2, 1, 1)
    for i in range(N_anchors):
        plt.plot(time, d1_x_hist[:, i], label=f'Anch {i}')
    plt.axvspan(fault_start, fault_end-1, lw=1, alpha=0.15, color='red', label='Faults')
    plt.ylabel('d1 (m)')
    plt.title('Optimal fault vector d1')
    plt.grid(True)
    plt.legend(ncol=6)
    plt.ylim(-1, 1.5)   

    plt.subplot(2, 1, 2)
    for i in range(N_anchors):
        plt.plot(time, d2_x_hist[:, i], label=f'Anch {i}')
    plt.axvspan(fault_start, fault_end-1, lw=1, alpha=0.15, color='red', label='Faults')
    plt.xlabel('Time step')
    plt.ylabel('d2 (m)')
    plt.title('Optimal fault vector d2')
    plt.grid(True)
    plt.legend(ncol=6)
    plt.ylim(-1, 1.5) 

    plt.tight_layout()
    plt.savefig('d1_d2.pdf', format='pdf')
    plt.show()

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
    cover_norm_new = np.all(error_norm <= PL_norm_new)

    print("\n=== PL Coverage Check (UNIFIED Quantiles) ===")
    print(f"Traditional PL (single fault):")
    print(f"  X: {cover_trad_x}, Y: {cover_trad_y}, Norm: {cover_norm_trad}")
    print(f"Section 2.7 PL (integer opt, corrected):")
    print(f"  X: {cover_new_x}, Y: {cover_new_y}, Norm: {cover_norm_new}")

    print(f"\nMean absolute error: X = {np.mean(np.abs(true_error[:,0])):.3f} m, "
          f"Y = {np.mean(np.abs(true_error[:,1])):.3f} m")
    print(f"Mean PL (traditional): X = {np.mean(PL_x_trad):.3f} m, Y = {np.mean(PL_y_trad):.3f} m")
    print(f"Mean PL (Section 2.7): X = {np.mean(PL_x_new):.3f} m, Y = {np.mean(PL_y_new):.3f} m")


if __name__ == "__main__":
    run_simulation()