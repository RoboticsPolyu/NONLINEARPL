"""
UWB Positioning Simulation with:
  - Time-varying random faults
  - Iterative FDE (chi-square residual test)
  - Advanced RAIM MHSS (HPL)
  - Traditional single-fault PL (Sec 2.5) - WITH UNIFIED QUANTILES
  - Section 2.7 multi-fault PL (integer optimisation) -
        * X/Y directions (original)
        * Radial (Euclidean norm) - NEW (using norm instead of e1 projection)
  - NONLINEAR WORST-CASE ANALYSIS (NEW) -
        * Gauss-Newton nonlinear solver
        * Worst-case fault pattern identification
        * Comparison with linear methods
All previous functionality is preserved; PL methods are added for comparison.
Fixed H-robustness issue.
"""

import numpy as np
from scipy.stats import chi2, ncx2, norm
from scipy.linalg import inv, pinv
from scipy.optimize import root_scalar, minimize
import itertools
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical symbols

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
angles = np.deg2rad([0, 60, 120, 180, 240, 300, 330])
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

def compute_jacobian(pos, anchors):
    """Compute the Jacobian matrix H for the measurement model."""
    m = len(anchors)
    H = np.zeros((m, 2))
    for i in range(m):
        diff = pos - anchors[i]
        norm_val = np.linalg.norm(diff)
        if norm_val > 1e-6:
            H[i, :] = diff / norm_val
    return H

def wls_position(measurements, anchors, W, max_iter=20, tol=1e-8):
    """
    Weighted Least-Squares via Gauss-Newton.
    Returns (position, Jacobian H). H is always a 2D array (m x 2) if m>=2.
    """
    m = len(anchors)
    if m < 2:
        return np.mean(anchors, axis=0), None
    x = np.mean(anchors, axis=0)
    H = None  # ensure defined even if loop fails
    for _ in range(max_iter):
        r = compute_ranges(x, anchors)
        residuals = measurements - r
        H = compute_jacobian(x, anchors)
        HtWH = H.T @ W @ H
        HtWres = H.T @ W @ residuals
        try:
            delta = inv(HtWH) @ HtWres
        except np.linalg.LinAlgError:
            break
        x += delta
        if np.linalg.norm(delta) < tol:
            break
    return x, H

def nonlinear_wls_position(measurements, anchors, W, max_iter=50, tol=1e-10):
    """
    Enhanced nonlinear WLS using Gauss-Newton with line search.
    This is the nonlinear solver for the worst-case analysis.
    """
    m = len(anchors)
    if m < 2:
        return np.mean(anchors, axis=0), None
    
    x = np.mean(anchors, axis=0)
    H = None
    
    for iteration in range(max_iter):
        r = compute_ranges(x, anchors)
        residuals = measurements - r
        H = compute_jacobian(x, anchors)
        
        HtWH = H.T @ W @ H
        HtWres = H.T @ W @ residuals
        
        try:
            delta = inv(HtWH) @ HtWres
        except np.linalg.LinAlgError:
            delta = pinv(HtWH) @ HtWres
        
        # Line search for robustness
        alpha = 1.0
        cost_old = residuals.T @ W @ residuals
        
        for _ in range(10):
            x_new = x + alpha * delta
            r_new = compute_ranges(x_new, anchors)
            res_new = measurements - r_new
            cost_new = res_new.T @ W @ res_new
            
            if cost_new < cost_old:
                x = x_new
                break
            alpha *= 0.5
        
        if np.linalg.norm(alpha * delta) < tol:
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
    H = compute_jacobian(pos_est, anchors)
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
    H_full = compute_jacobian(full_pos, anchors)

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

        H_sub = compute_jacobian(pos_sub, anch_sub)
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
# 6. Protection Level Calculators
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

# ------------------------------------------------------------
# NEW: Nonlinear Worst-Case Analysis (Based on Section 4 of the document)
# ------------------------------------------------------------
def compute_nonlinear_worst_case(measurements, anchors, W, R, D, alpha_pl, 
                                x_true=None, max_faults=2, max_iter=50):
    """
    Compute the worst-case state estimate under multiple faults using
    nonlinear Gauss-Newton optimization.
    
    This implements the worst-case analysis described in Section 4:
    For each fault pattern, solve the nonlinear WLS problem and find
    the pattern that maximizes the state estimation deviation.
    
    Parameters:
    -----------
    measurements : array (m,)
        Current measurements
    anchors : array (m, 2)
        Anchor positions
    W : array (m, m)
        Weighting matrix
    R : int
        Total number of faults
    D : float
        Maximum fault magnitude
    alpha_pl : float
        Integrity risk
    x_true : array (2,), optional
        True position for deviation calculation
    max_faults : int
        Maximum number of faults to consider (for combinations)
    max_iter : int
        Maximum iterations for nonlinear solver
    
    Returns:
    --------
    worst_pattern : array (m,)
        Fault allocation vector for worst case
    worst_b : float
        Optimal fault magnitude
    worst_x : array (2,)
        State estimate for worst case
    worst_deviation : float
        Maximum deviation
    all_results : list
        All (pattern, b, x, deviation) for analysis
    """
    n_meas = len(anchors)
    if n_meas < 2:
        return None, None, None, 0.0, []
    
    # 1. Generate all fault patterns (compositions of R into n parts)
    patterns = []
    for r in range(1, min(max_faults, n_meas) + 1):
        for combo in itertools.combinations(range(n_meas), r):
            lam = np.zeros(n_meas)
            for idx in combo:
                lam[idx] += 1
            patterns.append(lam)
    
    # Also include the no-fault pattern for reference
    patterns.append(np.zeros(n_meas))
    
    # 2. Discretize b for search (if not optimizing jointly)
    b_values = np.linspace(0, D, 21)  # 21 points for reasonable resolution
    
    # 3. Get nominal (fault-free) estimate
    x_nominal, H_nom = nonlinear_wls_position(measurements, anchors, W, max_iter=max_iter)
    
    # Store all results
    all_results = []
    worst_deviation = -np.inf
    worst_pattern = None
    worst_b = None
    worst_x = None
    
    print(f"\n--- Nonlinear Worst-Case Search ---")
    print(f"Total patterns: {len(patterns)}, b grid: {len(b_values)} points")
    
    # 4. For each pattern, solve nonlinear WLS for each b
    for idx, lam in enumerate(patterns):
        # Skip the no-fault pattern for worst-case search (unless it's the only one)
        if np.sum(lam) == 0:
            continue
            
        for b in b_values:
            if b < 1e-9:
                continue
                
            b_vec = b * lam
            
            # Solve nonlinear WLS with fault bias
            try:
                x_star, H_star = nonlinear_wls_position(
                    measurements - b_vec,  # Subtract fault bias from measurements
                    anchors, W, 
                    max_iter=max_iter
                )
            except Exception as e:
                continue
            
            # Compute deviation from nominal (or true if provided)
            if x_true is not None:
                deviation = np.linalg.norm(x_star - x_true)
            else:
                deviation = np.linalg.norm(x_star - x_nominal)
            
            # Record result
            all_results.append({
                'pattern': lam.copy(),
                'b': b,
                'b_vec': b_vec.copy(),
                'x': x_star.copy(),
                'deviation': deviation,
                'H': H_star
            })
            
            # Update worst case
            if deviation > worst_deviation:
                worst_deviation = deviation
                worst_pattern = lam.copy()
                worst_b = b
                worst_x = x_star.copy()
    
    # Sort results by deviation
    all_results.sort(key=lambda r: r['deviation'], reverse=True)
    
    print(f"Worst deviation: {worst_deviation:.4f} m")
    print(f"Worst pattern: {worst_pattern}")
    print(f"Worst b: {worst_b:.4f} m")
    if worst_x is not None:
        print(f"Worst x: ({worst_x[0]:.4f}, {worst_x[1]:.4f}) m")
    
    return worst_pattern, worst_b, worst_x, worst_deviation, all_results

def compute_nonlinear_worst_case_batch(measurements_seq, anchors, W, R, D, 
                                       alpha_pl, true_traj, max_faults=2):
    """
    Batch version: compute nonlinear worst-case for entire trajectory.
    """
    num_steps = len(measurements_seq)
    worst_deviations = np.zeros(num_steps)
    worst_patterns = []
    worst_xs = []
    worst_bs = []
    
    print(f"\n=== Nonlinear Worst-Case Batch Analysis ===")
    print(f"Processing {num_steps} time steps...")
    
    start_time = time.time()
    
    for k, meas in enumerate(measurements_seq):
        if k % 100 == 0:
            print(f"  Step {k}/{num_steps}")
        
        x_true = true_traj[k] if true_traj is not None else None
        
        worst_pat, worst_b, worst_x, worst_dev, results = compute_nonlinear_worst_case(
            meas, anchors, W, R, D, alpha_pl, 
            x_true=x_true, max_faults=max_faults
        )
        
        worst_deviations[k] = worst_dev if worst_dev > 0 else 0.0
        worst_patterns.append(worst_pat)
        worst_xs.append(worst_x)
        worst_bs.append(worst_b)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    
    return worst_deviations, worst_patterns, worst_xs, worst_bs

# ------------------------------------------------------------
# NEW: Comparison function for linear vs nonlinear worst-case
# ------------------------------------------------------------
def compute_linear_worst_case_sensitivity(H, W, R, D):
    """
    Compute linear worst-case based on sensitivity analysis.
    This implements the closed-form solution from Section 3.2.
    
    Returns the worst-case pattern and deviation.
    """
    n = H.shape[0]
    if n == 0:
        return None, 0.0
    
    try:
        P = inv(H.T @ W @ H)
        S = P @ H.T @ W  # Sensitivity matrix
    except np.linalg.LinAlgError:
        P = pinv(H.T @ W @ H)
        S = P @ H.T @ W
    
    # For Euclidean norm, worst case is all faults on most sensitive channel
    sensitivities = np.array([np.linalg.norm(S[:, i]) for i in range(n)])
    worst_idx = np.argmax(sensitivities)
    
    # Worst pattern: all R faults on the most sensitive channel
    worst_pattern = np.zeros(n)
    worst_pattern[worst_idx] = R
    
    # Worst deviation
    worst_deviation = D * sensitivities[worst_idx] * R
    
    return worst_pattern, worst_deviation, S

def compare_linear_vs_nonlinear_worst_case(measurements, anchors, W, R, D, alpha_pl):
    """
    Compare linear and nonlinear worst-case analysis.
    """
    print("\n" + "="*60)
    print("COMPARISON: Linear vs Nonlinear Worst-Case Analysis")
    print("="*60)
    
    # 1. Get nominal estimate and Jacobian
    x_nominal, H = nonlinear_wls_position(measurements, anchors, W)
    
    if H is None or H.shape[0] < 2:
        print("Insufficient measurements for comparison")
        return
    
    # 2. Linear worst-case (closed form)
    linear_pattern, linear_dev, S = compute_linear_worst_case_sensitivity(H, W, R, D)
    
    print("\n--- Linear Worst-Case (Closed Form) ---")
    print(f"Pattern: {linear_pattern}")
    print(f"Deviation bound: {linear_dev:.4f} m")
    print(f"Sensitivities per channel: {[np.linalg.norm(S[:, i]) for i in range(H.shape[0])]}")
    
    # 3. Nonlinear worst-case (exhaustive search)
    worst_pattern, worst_b, worst_x, nonlinear_dev, results = compute_nonlinear_worst_case(
        measurements, anchors, W, R, D, alpha_pl,
        x_true=x_nominal, max_faults=R
    )
    
    print("\n--- Nonlinear Worst-Case (Exhaustive) ---")
    print(f"Pattern: {worst_pattern}")
    print(f"b: {worst_b:.4f} m")
    print(f"Deviation: {nonlinear_dev:.4f} m")
    
    # 4. Compare
    print("\n--- Comparison ---")
    print(f"Linear deviation: {linear_dev:.4f} m")
    print(f"Nonlinear deviation: {nonlinear_dev:.4f} m")
    print(f"Ratio (nonlinear/linear): {nonlinear_dev/linear_dev:.3f}")
    
    # 5. Check if linear prediction matches nonlinear
    pattern_match = np.array_equal(linear_pattern, worst_pattern)
    print(f"Pattern match: {'Yes' if pattern_match else 'No'}")
    
    if not pattern_match:
        print("Note: Nonlinear effects cause different worst-case pattern!")
        print("The linear approximation may be conservative or optimistic.")
    
    return {
        'linear_pattern': linear_pattern,
        'linear_deviation': linear_dev,
        'nonlinear_pattern': worst_pattern,
        'nonlinear_b': worst_b,
        'nonlinear_deviation': nonlinear_dev,
        'pattern_match': pattern_match
    }

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
    seq = amplitude * np.ones(num_steps)
    seq[:fault_start] = 0.0
    seq[fault_end:] = 0.0
    return seq

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

    # --- NEW: Storage for chi-square test statistics ---
    sse_vals = np.full(num_steps, np.nan)
    threshold_vals = np.full(num_steps, np.nan)
    
    # --- NEW: Storage for nonlinear worst-case analysis ---
    nl_worst_deviations = np.zeros(num_steps)
    nl_worst_patterns = []
    nl_worst_xs = []
    nl_worst_bs = []

    # Storage for optimal d vectors (optional)
    d1_x_hist = np.full((num_steps, N_anchors), np.nan)
    d2_x_hist = np.full((num_steps, N_anchors), np.nan)
    d1_rad_hist = np.full((num_steps, N_anchors), np.nan)
    d2_rad_hist = np.full((num_steps, N_anchors), np.nan)

    # ---- Simulation loop ----
    print("\n=== Simulation Start ===")
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

        # WLS estimate and Jacobian (using nonlinear solver)
        pos_est, H = nonlinear_wls_position(kept_meas, kept_anchors, W_kept)

        # Robustness: if H is invalid, build it manually
        if H is None or not hasattr(H, 'shape') or H.ndim < 2 or H.shape[0] < 2:
            pos_est = np.mean(kept_anchors, axis=0)
            H = compute_jacobian(pos_est, kept_anchors)

        est_traj[k] = pos_est
        true_error[k] = pos_est - true_pos

        # Record excluded anchors (those not in keep_idx)
        excluded_ids = [i for i in range(N_anchors) if i not in keep_idx]
        excluded_history.append(excluded_ids)

        # Chi-square threshold for the kept subset
        dof_kept = n_kept - 2
        chi2_thr = chi2.ppf(1 - alpha, dof_kept) if dof_kept > 0 else 0.0

        # Compute SSE for the kept subset
        if n_kept >= 2:
            r = kept_meas - compute_ranges(pos_est, kept_anchors)
            sse = r @ W_kept @ r
        else:
            sse = np.nan
        sse_vals[k] = sse
        threshold_vals[k] = chi2_thr

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
        
        # ---- NEW: Nonlinear worst-case analysis (every 50 steps for efficiency) ----
        if k % 1 == 0 and n_kept >= 4:
            # Use a subset of measurements for worst-case analysis
            worst_pat, worst_b, worst_x, worst_dev, _ = compute_nonlinear_worst_case(
                kept_meas, kept_anchors, W_kept, R=2, D=0.8, 
                alpha_pl=alpha_pl, x_true=true_pos, max_faults=2
            )
            nl_worst_deviations[k] = worst_dev if worst_dev is not None else 0.0
            nl_worst_patterns.append(worst_pat)
            nl_worst_xs.append(worst_x)
            nl_worst_bs.append(worst_b)
        else:
            # Interpolate for skipped steps
            if k > 0:
                nl_worst_deviations[k] = nl_worst_deviations[k-1]
            else:
                nl_worst_deviations[k] = 0.0

        if k % 100 == 0:
            print(f"  Step {k}/{num_steps}")

    # ---- Ensure excluded_history length matches num_steps ----
    if len(excluded_history) > num_steps:
        print(f"Warning: excluded_history length {len(excluded_history)} > num_steps {num_steps}. Trimming.")
        excluded_history = excluded_history[:num_steps]
    elif len(excluded_history) < num_steps:
        print(f"Warning: excluded_history length {len(excluded_history)} < num_steps {num_steps}. Padding with empty lists.")
        excluded_history.extend([[] for _ in range(num_steps - len(excluded_history))])

    # ---- Post-processing and plots ----
    time = np.arange(num_steps)
    error_norm = np.linalg.norm(true_error, axis=1)

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
    ax2a.plot(time, PL_x_new, 'g-', lw=1.0, label='PL X')
    ax2a.plot(time, abs(true_error[:, 0]), 'b.-', lw=0.4, label='|Error X|')
    ax2a.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax2a.set_ylabel('Error / PL (m)')
    ax2a.legend(fontsize=6, loc='upper left')
    ax2a.grid(True)
    ax2a.set_title('X-direction', fontsize=8)
    # Subplot 2: Y-direction
    ax2b.plot(time, PL_y_new, 'g-', lw=1.0, label='PL Y')
    ax2b.plot(time, abs(true_error[:, 1]), 'b.-', lw=0.4, label='|Error Y|')
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
    ax3.plot(time, PL_radial_new, 'g-', lw=1.2, label='Radial PL')
    ax3.plot(time, hpl_vals, 'm--', lw=0.8, label='ARAIM HPL')
    ax3.plot(time, error_norm, 'b.-', lw=0.4, label='|Error| (Euclidean)')
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

    # --------------------------------------------------
    # Figure 4: Chi-square test, threshold, and detection analysis (CORRECTED)
    # --------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(6.8, 4.5), sharex=True)

    # Upper: SSE and threshold (unchanged)
    ax4a.plot(time, sse_vals, 'b-', lw=1.0, label='SSE (χ² statistic)')
    ax4a.plot(time, threshold_vals, 'r--', lw=1.0, label='Threshold')
    ax4a.axvspan(fault_start, fault_end-1, alpha=0.15, color='red', label='Fault active')
    ax4a.set_ylabel('χ² value')
    ax4a.legend(fontsize=6, loc='upper left')
    ax4a.grid(True)
    ax4a.set_title('Chi‑square test statistic and threshold (FDE)', fontsize=8)

    # Lower: Detection results with time‑aware logic
    corr_times = []
    corr_ids = []
    false_times = []
    false_ids = []
    miss_times = []
    miss_ids = []

    for k in range(num_steps):
        # 1) Collect exclusions at this time step
        excl_ids = excluded_history[k] if k < len(excluded_history) else []
        fault_on = fault_active[k]

        # 2) Classify each excluded anchor
        for idx in excl_ids:
            if fault_on and (idx in fault_indices):
                # Correct detection: fault active AND anchor is faulty
                corr_times.append(k)
                corr_ids.append(idx)
            else:
                # False alarm: either no fault OR anchor is not faulty
                false_times.append(k)
                false_ids.append(idx)

        # 3) Missed detections: faulty anchors NOT excluded during fault
        if fault_on:
            for idx in fault_indices:
                if idx not in excl_ids:
                    miss_times.append(k)
                    miss_ids.append(idx)

    # Plot the three categories
    if corr_times:
        ax4b.scatter(corr_times, corr_ids, s=1, c='green', marker='o', alpha=0.7, label='Correct detection')
    if false_times:
        ax4b.scatter(false_times, false_ids, s=5, c='red', marker='x', alpha=0.7, label='False alarm')
    if miss_times:
        ax4b.scatter(miss_times, miss_ids, s=5, c='blue', marker='.', alpha=0.7, label='Missed detection')

    ax4b.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax4b.set_xlabel('Time step')
    ax4b.set_ylabel('Anchor index')
    ax4b.set_yticks(range(N_anchors))
    ax4b.grid(True, axis='y', linestyle=':')
    ax4b.legend(fontsize=6, loc='upper right')
    ax4b.set_title('Detection status of excluded anchors and missed faults (time‑aware)', fontsize=8)

    fig4.tight_layout()
    fig4.savefig('fig_chi2_exclusion.pdf', format='pdf')
    plt.close(fig4)

    # --------------------------------------------------
    # Figure 5: NEW - Nonlinear Worst-Case Analysis Results
    # --------------------------------------------------
    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(6.8, 4.5), sharex=True)
    
    # Upper: Nonlinear worst-case deviation vs actual error
    ax5a.plot(time, nl_worst_deviations, 'r-', lw=1.2, label='NL Worst-Case Deviation')
    ax5a.plot(time, error_norm, 'b.-', lw=0.4, label='|Error|')
    ax5a.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax5a.set_ylabel('Deviation (m)')
    ax5a.legend(fontsize=6, loc='upper left')
    ax5a.grid(True)
    ax5a.set_title('Nonlinear Worst-Case Deviation vs Actual Error', fontsize=8)
    
    # Lower: Ratio of nonlinear worst-case to actual error
    ratio = np.divide(nl_worst_deviations, error_norm + 1e-10)
    ax5b.plot(time, ratio, 'k-', lw=0.8, label='NL Worst / |Error|')
    ax5b.axhline(y=1.0, color='r', linestyle=':', lw=0.8, label='Ratio = 1')
    ax5b.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax5b.set_xlabel('Time step')
    ax5b.set_ylabel('Ratio')
    ax5b.legend(fontsize=6, loc='upper left')
    ax5b.grid(True)
    ax5b.set_title('Worst-Case to Actual Error Ratio', fontsize=8)
    
    fig5.tight_layout()
    fig5.savefig('fig_nonlinear_worstcase.pdf', format='pdf')
    plt.close(fig5)

    # --------------------------------------------------
    # Figure 6: Comparison of PL methods with nonlinear worst-case
    # --------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=(6.8, 3.5))
    ax6.plot(time, PL_radial_new, 'g-', lw=1.0, label='Radial PL (linear)')
    ax6.plot(time, nl_worst_deviations, 'r--', lw=1.0, label='NL Worst-Case')
    ax6.plot(time, error_norm, 'b.-', lw=0.4, label='|Error|', alpha=0.5)
    ax6.axvspan(fault_start, fault_end-1, alpha=0.15, color='red')
    ax6.set_xlabel('Time step')
    ax6.set_ylabel('Value (m)')
    ax6.legend(fontsize=6, loc='upper left')
    ax6.grid(True)
    ax6.set_title('Comparison: Linear PL vs Nonlinear Worst-Case', fontsize=8)
    fig6.tight_layout()
    fig6.savefig('fig_pl_vs_nonlinear.pdf', format='pdf')
    plt.close(fig6)

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

    # FDE exclusions
    excluded_counts = [len(e) for e in excluded_history]
    print(f"\nFDE exclusions: mean {np.mean(excluded_counts):.2f}, max {np.max(excluded_counts)}")

    # Safe computation of fault_excl
    fault_excl = [excl for k, excl in enumerate(excluded_history) if k < num_steps and fault_active[k]]
    if fault_excl:
        print(f"Avg excluded anchors during faults: {np.mean([len(e) for e in fault_excl]):.2f}")

    # Detection performance metrics (correct, false, missed)
    total_corr = len(corr_times)
    total_false = len(false_times)
    total_miss = len(miss_times)
    print("\n=== Detection Performance (during fault active period) ===")
    print(f"Correct exclusions (faulty anchor removed during fault): {total_corr}")
    print(f"False alarms (non‑faulty anchor removed OR removal outside fault): {total_false}")
    print(f"Missed detections (faulty anchor not removed during fault): {total_miss}")
    if fault_steps > 0:
        total_possible = len(fault_indices) * fault_steps
        print(f"Missed detection rate: {total_miss / total_possible:.2%}")

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
    cover_radial_new_pct = np.mean(error_norm <= PL_radial_new) * 100
    mean_hpl = np.mean(hpl_vals)
    max_hpl = np.max(hpl_vals)
    mean_pl_radial = np.mean(PL_radial_new)
    max_pl_radial = np.max(PL_radial_new)

    print("\n=== Radial PL vs ARAIM HPL (Euclidean norm) ===")
    print(f"ARAIM HPL: Coverage = {cover_araim_radial:.1f}%, Mean = {mean_hpl:.3f} m, Max = {max_hpl:.3f} m")
    print(f"Section 2.7 Radial PL: Coverage = {cover_radial_new_pct:.1f}%, Mean = {mean_pl_radial:.3f} m, Max = {max_pl_radial:.3f} m")

    # ---- NEW: Nonlinear worst-case statistics ----
    valid_nl = nl_worst_deviations > 0
    if np.any(valid_nl):
        print("\n=== Nonlinear Worst-Case Analysis ===")
        print(f"Mean NL worst-case deviation: {np.mean(nl_worst_deviations[valid_nl]):.3f} m")
        print(f"Max NL worst-case deviation: {np.max(nl_worst_deviations):.3f} m")
        print(f"Mean ratio (NL worst / actual error): {np.mean(ratio[valid_nl]):.2f}")
        print(f"Coverage (NL worst >= actual error): {np.mean(nl_worst_deviations >= error_norm) * 100:.1f}%")

    # ---- Run a one-time comparison at a specific time step ----
    print("\n" + "="*60)
    print("DETAILED COMPARISON AT TIME STEP 300")
    print("="*60)
    
    # Get data at step 300
    k_compare = min(300, num_steps - 1)
    keep_idx, _ = detect_and_exclude(measurements, anchors, W_full, alpha, min_anchors=4)
    kept_anchors = anchors[keep_idx]
    kept_meas = measurements[keep_idx]
    W_kept = np.diag(1.0 / sigma**2 * np.ones(len(keep_idx)))
    
    # Run comparison
    comparison_results = compare_linear_vs_nonlinear_worst_case(
        kept_meas, kept_anchors, W_kept, R=2, D=1.0, alpha_pl=alpha_pl
    )


if __name__ == "__main__":
    run_simulation()