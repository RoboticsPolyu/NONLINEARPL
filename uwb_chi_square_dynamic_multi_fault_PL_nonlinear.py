"""
UWB Positioning Simulation with:
  - Time‑varying random faults
  - Iterative FDE (chi‑square residual test)
  - ARAIM MHSS (HPL)
  - Traditional single‑fault PL (Sec 2.5) - WITH UNIFIED QUANTILES
  - Corrected Section 2.7 multi‑fault PL (integer optimisation) - WITH UNIFIED QUANTILES
  - **NEW: Full bilevel nonlinear optimisation for Section 2.7 (sporadic application)**
All previous functionality is preserved; PL methods are added for comparison.
Fixed H‑robustness issue.
"""

import numpy as np
from scipy.stats import chi2, ncx2, norm
from scipy.linalg import inv, pinv
from scipy.optimize import root_scalar, minimize, Bounds
import itertools
import matplotlib.pyplot as plt
import time

# ------------------------------
# 1. System Configuration
# ------------------------------
np.random.seed(42)          # change or comment for fully random runs

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
alpha_pl = 1e-4       # integrity risk for ARAIM and ALL PL methods (UNIFIED)
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

def wls_position(measurements, anchors, W, max_iter=20):
    """
    Weighted Least‑Squares via Gauss‑Newton for nonlinear range model.
    Returns (position, Jacobian H) or (None,None) on failure.
    """
    m = len(anchors)
    if m < 2:
        return np.mean(anchors, axis=0), None
    x = np.mean(anchors, axis=0)
    H = None
    for _ in range(max_iter):
        r = compute_ranges(x, anchors)
        residuals = measurements - r
        H = np.zeros((m, state_dim))
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
def detect_and_exclude(measurements, anchors, W, alpha, min_anchors=4):
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
# 5. ARAIM MHSS
# ------------------------------
def compute_araim_pl(measurements, anchors, W, alpha_pl, AL):
    N = len(anchors)
    full_pos = wls_position(measurements, anchors, W)[0]  # ignore H
    H_full = np.zeros((N, 2))
    for i in range(N):
        diff = full_pos - anchors[i]
        norm_val = np.linalg.norm(diff)
        if norm_val > 1e-6:
            H_full[i, :] = diff / norm_val
    try:
        P_full = inv(H_full.T @ W @ H_full)
    except np.linalg.LinAlgError:
        P_full = np.eye(2) * 1e6

    subsets = []
    subsets.append(('full', list(range(N)), full_pos, P_full))

    for excl in range(N):
        keep = [i for i in range(N) if i != excl]
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

    M = len(subsets)
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
    """Traditional single‑fault PL (Section 2.5) with unified quantile."""
    n = H.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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

def compute_pl_section27_linear(H, W, chi2_threshold, max_faults, alpha_pl):
    """
    Linearised Section 2.7 PL (original) – returns d1,d2 and bias terms.
    (FIXED: removed the erroneous /2 factor)
    """
    n = H.shape[0]
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                None, None, None, None)
    total_hyp = 0
    for r in range(1, max_faults + 1):
        if r <= n:
            total_hyp += len(list(itertools.combinations(range(n), r)))
    if total_hyp == 0:
        total_hyp = 1
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
            bias_x = 2 * D_val * sum_abs_Mx   # CORRECTED: removed /2
            if bias_x > best_fault_x:
                best_fault_x = bias_x
                d1 = np.zeros(n); d2 = np.zeros(n)
                for idx in combo:
                    sign = np.sign(Mx[idx]) if abs(Mx[idx]) > 1e-12 else 0.0
                    d1[idx] = -D_val * sign
                    d2[idx] = +D_val * sign
                best_d1_x = d1
                best_d2_x = d2
            # Y direction
            sum_abs_My = np.sum([np.abs(My[i]) for i in combo])
            bias_y = 2 * D_val * sum_abs_My
            if bias_y > best_fault_y:
                best_fault_y = bias_y
                d1 = np.zeros(n); d2 = np.zeros(n)
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
# 7. NEW: Full Bilevel Nonlinear Optimisation for Section 2.7
# ------------------------------
def inner_wls_nonlinear(d, meas, anchors, W):
    """Solve min_x ||(z - d) - h(x)||^2_W. Returns position."""
    pos, _ = wls_position(meas - d, anchors, W)
    return pos

def compute_pl_section27_nonlinear(meas, anchors, W, chi2_threshold, max_faults, alpha_pl):
    """
    Full bilevel nonlinear optimisation for Section 2.7.
    For each fault pattern (combo), we solve:
        max_{d1,d2 in D} e_x^T [x^*(d1) - x^*(d2)]
    where x^*(d) = argmin_x ||(z-d)-h(x)||^2_W.
    We use scipy.optimize.minimize for the outer maximisation.
    This is extremely slow – only for demonstration.
    Returns the maximum bias and the optimal d1,d2 (as full vectors).
    """
    n = len(anchors)
    if n < 2:
        return 0.0, None, None, 0.0, None, None, 0.0, 0.0

    total_hyp = 0
    for r in range(1, max_faults + 1):
        if r <= n:
            total_hyp += len(list(itertools.combinations(range(n), r)))
    if total_hyp == 0:
        total_hyp = 1
    k_noise = norm.ppf(1 - alpha_pl / total_hyp)

    # Compute noise PL using linearised covariance (best we can do without Hessian)
    pos0, H0 = wls_position(meas, anchors, W)
    if H0 is None or H0.shape[0] < 2:
        return 0.0, None, None, 0.0, None, None, 0.0, 0.0
    try:
        P = inv(H0.T @ W @ H0)
    except np.linalg.LinAlgError:
        P = pinv(H0.T @ W @ H0)
    PL_noise_x = k_noise * np.sqrt(P[0, 0])
    PL_noise_y = k_noise * np.sqrt(P[1, 1])

    best_bias_x = 0.0
    best_bias_y = 0.0
    best_d1_x = None
    best_d2_x = None
    best_d1_y = None
    best_d2_y = None

    # For each fault combination
    for r in range(1, max_faults + 1):
        if r > n:
            break
        for combo in itertools.combinations(range(n), r):
            # Compute D bound from linearised residual (as before)
            try:
                HtWH_inv = inv(H0.T @ W @ H0)
            except np.linalg.LinAlgError:
                HtWH_inv = pinv(H0.T @ W @ H0)
            S = W - W @ H0 @ HtWH_inv @ H0.T @ W
            S_sub = S[np.ix_(combo, combo)]
            sum_abs_S = np.sum(np.abs(S_sub))
            if sum_abs_S < 1e-12:
                continue
            D_val = np.sqrt(chi2_threshold / sum_abs_S)

            # Define the feasible box for d (only combo indices can be non‑zero)
            lb = [-D_val] * (2*r)
            ub = [D_val] * (2*r)
            bounds = Bounds(lb, ub)
            x0 = np.zeros(2*r)

            # ---- X direction ----
            def objective_x(vec):
                d1 = np.zeros(n)
                d2 = np.zeros(n)
                for i, idx in enumerate(combo):
                    d1[idx] = vec[i]
                    d2[idx] = vec[r + i]
                x1 = inner_wls_nonlinear(d1, meas, anchors, W)
                x2 = inner_wls_nonlinear(d2, meas, anchors, W)
                return -(x1[0] - x2[0])

            res = minimize(objective_x, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 50, 'ftol': 1e-6})
            if res.success:
                d1_opt = np.zeros(n)
                d2_opt = np.zeros(n)
                for i, idx in enumerate(combo):
                    d1_opt[idx] = res.x[i]
                    d2_opt[idx] = res.x[r + i]
                x1_opt = inner_wls_nonlinear(d1_opt, meas, anchors, W)
                x2_opt = inner_wls_nonlinear(d2_opt, meas, anchors, W)
                bias_x = np.abs(x1_opt[0] - x2_opt[0])
                if bias_x > best_bias_x:
                    best_bias_x = bias_x
                    best_d1_x = d1_opt
                    best_d2_x = d2_opt

            # ---- Y direction ----
            def objective_y(vec):
                d1 = np.zeros(n); d2 = np.zeros(n)
                for i, idx in enumerate(combo):
                    d1[idx] = vec[i]
                    d2[idx] = vec[r + i]
                x1 = inner_wls_nonlinear(d1, meas, anchors, W)
                x2 = inner_wls_nonlinear(d2, meas, anchors, W)
                return -(x1[1] - x2[1])

            res_y = minimize(objective_y, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 50, 'ftol': 1e-6})
            if res_y.success:
                d1_opt = np.zeros(n); d2_opt = np.zeros(n)
                for i, idx in enumerate(combo):
                    d1_opt[idx] = res_y.x[i]
                    d2_opt[idx] = res_y.x[r + i]
                x1_opt = inner_wls_nonlinear(d1_opt, meas, anchors, W)
                x2_opt = inner_wls_nonlinear(d2_opt, meas, anchors, W)
                bias_y = np.abs(x1_opt[1] - x2_opt[1])
                if bias_y > best_bias_y:
                    best_bias_y = bias_y
                    best_d1_y = d1_opt
                    best_d2_y = d2_opt

    PL_x = PL_noise_x + best_bias_x
    PL_y = PL_noise_y + best_bias_y
    return (PL_x, PL_y, PL_noise_x, PL_noise_y, best_bias_x, best_bias_y,
            best_d1_x, best_d2_x, best_d1_y, best_d2_y)

# ------------------------------
# 8. Helper: generate time‑varying faults
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
# 9. Main Simulation
# ------------------------------
def run_simulation():
    num_steps = 1000
    t = np.linspace(0, 2 * np.pi, num_steps)
    true_traj = np.column_stack((5 + 6 * np.cos(t), 5 + 4 * np.sin(2 * t)))

    # Fault config (unchanged)
    min_faults = 2
    max_faults = 3
    fault_start, fault_end = 150, 800
    num_faults = np.random.randint(min_faults, max_faults + 1)
    fault_indices = np.random.choice(N_anchors, size=num_faults, replace=False)
    fault_bias_sequences = {}
    fault_params = {}
    for idx in fault_indices:
        amplitude = np.random.uniform(0.3, 0.9) * np.random.choice([-1, 1])
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
    hpl_vals = np.zeros(num_steps)
    alarm_araim = np.zeros(num_steps, dtype=bool)
    PL_x_trad = np.zeros(num_steps)
    PL_y_trad = np.zeros(num_steps)
    PL_x_new_lin = np.zeros(num_steps)   # linearised Section 2.7
    PL_y_new_lin = np.zeros(num_steps)
    PL_x_new_nonlin = np.full(num_steps, np.nan)   # nonlinear (only for some steps)
    PL_y_new_nonlin = np.full(num_steps, np.nan)
    fault_active = np.zeros(num_steps, dtype=bool)
    excluded_history = []

    # Storage for d1,d2 (linear)
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

        dof_kept = n_kept - 2
        chi2_thr = chi2.ppf(1 - alpha, dof_kept) if dof_kept > 0 else 0.0

        # ARAIM
        hpl, alarm, _ = compute_araim_pl(measurements, anchors, W_full, alpha_pl, AL)
        hpl_vals[k] = hpl
        alarm_araim[k] = alarm

        # Traditional PL
        PL_x_trad[k], PL_y_trad[k], _, _, _, _ = compute_pl_traditional(
            H, W_kept, chi2_thr, alpha_pl
        )

        # Linearised Section 2.7 (fixed /2 bug)
        (PL_x_new_lin[k], PL_y_new_lin[k], _, _, _, _,
         d1_x, d2_x, d1_y, d2_y) = compute_pl_section27_linear(
            H, W_kept, chi2_thr, max_faults=2, alpha_pl=alpha_pl
        )

        # Store d1_x, d2_x for visualisation
        if d1_x is not None:
            d1_full = np.zeros(N_anchors)
            d2_full = np.zeros(N_anchors)
            for i, idx in enumerate(keep_idx):
                d1_full[idx] = d1_x[i]
                d2_full[idx] = d2_x[i]
            d1_x_hist[k] = d1_full
            d2_x_hist[k] = d2_full

        # ---- Apply full nonlinear bilevel optimisation only for every 10th step ----
        if k % 1 == 0 and n_kept >= 2:
            try:
                (PL_x_nl, PL_y_nl, _, _, _, _,
                 d1_nl_x, d2_nl_x, d1_nl_y, d2_nl_y) = compute_pl_section27_nonlinear(
                    kept_meas, kept_anchors, W_kept, chi2_thr, max_faults=2, alpha_pl=alpha_pl
                )
                PL_x_new_nonlin[k] = PL_x_nl
                PL_y_new_nonlin[k] = PL_y_nl
            except Exception as e:
                # In case of numerical failure, leave NaN
                pass

    # ---- Post‑processing and plots ----
    time = np.arange(num_steps)
    error_norm = np.linalg.norm(true_error, axis=1)

    # Plot 1: Trajectory (unchanged)
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

    # Plot 2: PL comparison (include nonlinear points)
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(time, abs(true_error[:, 0]), 'b-', lw=1.5, label='Error X')
    plt.plot(time, PL_x_trad, 'r-', lw=2, label='Trad PL X')
    plt.plot(time, PL_x_new_lin, 'g-', lw=2, label='Sec2.7 Lin X')
    valid_nl = ~np.isnan(PL_x_new_nonlin)
    if np.any(valid_nl):
        plt.scatter(time[valid_nl], PL_x_new_nonlin[valid_nl], c='orange', s=30, label='Sec2.7 Nonlin X')
    plt.axvspan(fault_start, fault_end-1, alpha=0.15, color='red', label='Faults')
    plt.ylabel('Error / PL (m)'); plt.legend(); plt.grid(True)
    plt.title('X‑direction: Error vs. Protection Levels (including Nonlinear)')

    plt.subplot(2, 1, 2)
    plt.plot(time, abs(true_error[:, 1]), 'b-', lw=1.5, label='Error Y')
    plt.plot(time, PL_y_trad, 'r-', lw=2, label='Trad PL Y')
    plt.plot(time, PL_y_new_lin, 'g-', lw=2, label='Sec2.7 Lin Y')
    valid_nl_y = ~np.isnan(PL_y_new_nonlin)
    if np.any(valid_nl_y):
        plt.scatter(time[valid_nl_y], PL_y_new_nonlin[valid_nl_y], c='orange', s=30, label='Sec2.7 Nonlin Y')
    plt.axvspan(fault_start, fault_end-1, alpha=0.15, color='red', label='Faults')
    plt.xlabel('Time step'); plt.ylabel('Error / PL (m)'); plt.legend(); plt.grid(True)
    plt.title('Y‑direction: Error vs. Protection Levels (including Nonlinear)')
    plt.tight_layout()

    # Plot 3: d1_x and d2_x over time (to verify correctness)
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    for i in range(N_anchors):
        plt.plot(time, d1_x_hist[:, i], label=f'Anch {i}')
    plt.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    plt.ylabel('d1_x (m)')
    plt.title('Optimal fault vector d1_x (linear)')
    plt.grid(True); plt.legend(ncol=3)

    plt.subplot(2, 1, 2)
    for i in range(N_anchors):
        plt.plot(time, d2_x_hist[:, i], label=f'Anch {i}')
    plt.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    plt.xlabel('Time step')
    plt.ylabel('d2_x (m)')
    plt.title('Optimal fault vector d2_x (linear)')
    plt.grid(True); plt.legend(ncol=3)
    plt.tight_layout()
    plt.show()

    # ---- Statistics ----
    fault_steps = np.sum(fault_active)
    print("\n=== Performance Summary ===")
    print(f"Faulty anchors: {fault_indices}")
    print(f"Fault duration: steps {fault_start} to {fault_end-1}")
    print(f"Steps with faults: {fault_steps}")

    total_araim_alarms = np.sum(alarm_araim)
    false_araim = np.sum(alarm_araim & ~fault_active)
    missed_araim = np.sum(~alarm_araim & fault_active)
    print(f"\nARAIM (HPL > {AL} m):")
    print(f"  Alarms raised: {total_araim_alarms}")
    print(f"  False alarms: {false_araim}")
    print(f"  Missed detections: {missed_araim}")
    if fault_steps > 0:
        print(f"  Detection rate: {1 - missed_araim/fault_steps:.3f}")
    print(f"  Mean HPL: {np.mean(hpl_vals):.3f} m, Max HPL: {np.max(hpl_vals):.3f} m")

    excluded_counts = [len(e) for e in excluded_history]
    print(f"\nFDE exclusions: mean {np.mean(excluded_counts):.2f}, max {np.max(excluded_counts)}")

    # PL coverage (using linear PL for simplicity)
    PL_norm_trad = np.sqrt(PL_x_trad**2 + PL_y_trad**2)
    PL_norm_lin = np.sqrt(PL_x_new_lin**2 + PL_y_new_lin**2)
    cover_trad_x = np.all(np.abs(true_error[:, 0]) <= PL_x_trad)
    cover_trad_y = np.all(np.abs(true_error[:, 1]) <= PL_y_trad)
    cover_lin_x = np.all(np.abs(true_error[:, 0]) <= PL_x_new_lin)
    cover_lin_y = np.all(np.abs(true_error[:, 1]) <= PL_y_new_lin)
    print("\n=== PL Coverage Check (Linear Section 2.7) ===")
    print(f"Traditional PL: X={cover_trad_x}, Y={cover_trad_y}")
    print(f"Section 2.7 Linear: X={cover_lin_x}, Y={cover_lin_y}")

if __name__ == "__main__":
    run_simulation()