"""
UWB Positioning Simulation with 6 Anchors, Time‑Varying Faults,
and ARAIM (Advanced RAIM) with MHSS and Protection Levels.
Includes MDB computation and iterative FDE for comparison.

The biases are not constant but follow a sinusoidal sequence
over the fault interval, simulating slowly drifting errors.
"""

import numpy as np
from scipy.stats import chi2, ncx2, norm
from scipy.linalg import inv, pinv
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# ------------------------------
# 1. System Configuration
# ------------------------------
np.random.seed(42)          # change or comment for fully random runs

center = np.array([5.0, 5.0])
radius = 10.0
angles = np.deg2rad([0, 30, 60, 120, 180, 240, 300])
anchors = np.array([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])
N_anchors = anchors.shape[0]
state_dim = 2
sigma = 0.1
W_full = np.diag(1.0 / sigma**2 * np.ones(N_anchors))

# Integrity parameters
alpha = 0.05          # false alarm rate for FDE
beta = 0.8            # detection probability for MDB
alpha_pl = 1e-4       # integrity risk for ARAIM
AL = 1.5              # alert limit (meters)

# ------------------------------
# 2. MDB Computation (Corrected)
# ------------------------------
def compute_mdb(anchors, W, alpha=0.05, beta=0.8):
    """
    Compute Minimum Detectable Bias (MDB) for each anchor.
    Uses the residual sensitivity matrix S = W - W H (H^T W H)^-1 H^T W.
    Returns mdb array, diagonal of S, and the non‑centrality parameter.
    """
    N = anchors.shape[0]
    df = N - state_dim
    threshold = chi2.ppf(1 - alpha, df)

    # Linearisation point (centre of anchors)
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

    # Find non‑centrality parameter lambda such that detection prob = beta
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
    """Euclidean distances from pos to all anchors."""
    return np.sqrt(np.sum((anchors - pos)**2, axis=1))

def wls_position(measurements, anchors, W):
    """
    Weighted Least‑Squares position estimate via Gauss‑Newton.
    Returns the estimated position.
    """
    if len(anchors) < 2:
        return np.mean(anchors, axis=0)
    x = np.mean(anchors, axis=0)
    for _ in range(10):
        r = compute_ranges(x, anchors)
        residuals = measurements - r
        H = np.zeros((len(anchors), 2))
        for i in range(len(anchors)):
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
    return x

def compute_wsse(measurements, pos_est, anchors, W):
    """Weighted Sum of Squared Errors."""
    r_est = compute_ranges(pos_est, anchors)
    e = measurements - r_est
    return e.T @ W @ e

def compute_standardised_residuals(measurements, pos_est, anchors, W):
    """
    Compute standardised residuals for each measurement.
    std_res_i = |e_i| / sqrt( (Q_res)_ii )
    where Q_res = W^-1 - H (H^T W H)^-1 H^T.
    """
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
# 4. Iterative Fault Detection and Exclusion (FDE)
# ------------------------------
def wls_position_with_iterative_fde(measurements, anchors, W_full, alpha, min_anchors = 4):
    """
    Iterative FDE: remove the anchor with the largest standardised residual
    until the chi‑square test passes or too few anchors remain.
    Returns: (position, list_of_excluded_indices, number_excluded)
    """
    excluded = []
    max_iter = len(anchors) - min_anchors

    for _ in range(max_iter):
        keep = [i for i in range(len(anchors)) if i not in excluded]
        m = len(keep)
        if m < min_anchors:
            break
        meas_sub = measurements[keep]
        anch_sub = anchors[keep]
        W_sub = W_full[keep][:, keep]

        pos_est = wls_position(meas_sub, anch_sub, W_sub)
        wsse = compute_wsse(meas_sub, pos_est, anch_sub, W_sub)
        dof = m - state_dim
        threshold = chi2.ppf(1 - alpha, dof) if dof > 0 else 0.0

        if wsse <= threshold:
            return pos_est, excluded, len(excluded)

        std_res = compute_standardised_residuals(meas_sub, pos_est, anch_sub, W_sub)
        worst_local = np.argmax(std_res)
        worst_global = keep[worst_local]
        excluded.append(worst_global)

    # Fallback: if too few remain, use all anchors (or the remaining subset)
    keep = [i for i in range(len(anchors)) if i not in excluded]
    if len(keep) >= 2:
        pos_est = wls_position(measurements[keep], anchors[keep], W_full[keep][:, keep])
    else:
        pos_est = np.mean(anchors, axis=0)
    return pos_est, excluded, len(excluded)

# ------------------------------
# 5. ARAIM: MHSS and Protection Level
# ------------------------------
def compute_araim_pl(measurements, anchors, W, alpha_pl, AL):
    """
    ARAIM MHSS (Multiple Hypothesis Solution Separation):
    - Full set and N single‑fault subsets.
    - Allocates integrity risk equally among hypotheses.
    Returns: (hpl, alarm_flag, subset_solutions)
    """
    N = len(anchors)

    # ---- Full solution ----
    full_pos = wls_position(measurements, anchors, W)
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

    # ---- Single‑fault subsets ----
    for excl in range(N):
        keep = [i for i in range(N) if i != excl]
        if len(keep) < state_dim + 1:   # need at least 3 anchors
            continue
        meas_sub = measurements[keep]
        anch_sub = anchors[keep]
        W_sub = W[keep][:, keep]
        pos_sub = wls_position(meas_sub, anch_sub, W_sub)

        # Covariance for this subset
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
    k_md = norm.ppf(1 - alpha_pl / M)   # quantile for fault‑free missed detection

    hpl_list = []
    for _, keep, pos_sub, P_sub in subsets:
        # Horizontal standard deviation (major axis)
        eigenvals = np.linalg.eigvalsh(P_sub)
        sigma_h = np.sqrt(np.max(eigenvals))
        # Bias = difference between subset and full solution
        bias = np.linalg.norm(pos_sub - full_pos)
        hpl_i = k_md * sigma_h + bias
        hpl_list.append(hpl_i)

    hpl = max(hpl_list)
    alarm = hpl > AL
    return hpl, alarm, subsets

# ------------------------------
# 6. Helper: inject time‑varying faults
# ------------------------------
def generate_bias_sequence(num_steps, fault_start, fault_end,
                           amplitude, frequency, phase):
    """
    Generate a sinusoidal bias sequence that is active only
    between fault_start and fault_end.
    """
    t = np.arange(num_steps)
    # sine wave: amplitude * sin(2π * freq * (t - fault_start) / duration + phase)
    duration = fault_end - fault_start
    if duration <= 0:
        return np.zeros(num_steps)
    seq = amplitude * np.sin(2 * np.pi * frequency * (t - fault_start) / duration + phase)
    seq[:fault_start] = 0.0
    seq[fault_end:] = 0.0
    return seq

# ------------------------------
# 7. Main Simulation with Random Time‑Varying Faults
# ------------------------------
def run_trajectory_simulation():
    num_steps = 1000
    t = np.linspace(0, 2 * np.pi, num_steps)
    true_traj = np.column_stack((5 + 6 * np.cos(t), 5 + 4 * np.sin(2 * t)))

    # ---- Random fault configuration (time‑varying) ----
    min_faults = 2
    max_faults = 3
    fault_start, fault_end = 150, 800

    num_faults = np.random.randint(min_faults, max_faults + 1)
    fault_indices = np.random.choice(N_anchors, size=num_faults, replace=False)

    # For each faulty anchor, generate a sinusoidal bias sequence
    fault_bias_sequences = {}
    fault_params = {}   # store amplitude & frequency for display

    for idx in fault_indices:
        amplitude = np.random.uniform(0.5, 0.9) * np.random.choice([-1, 1])
        frequency = np.random.uniform(0.5, 2.0)      # cycles per fault duration
        phase = np.random.uniform(0, 2 * np.pi)
        seq = generate_bias_sequence(num_steps, fault_start, fault_end,
                                     amplitude, frequency, phase)
        fault_bias_sequences[idx] = seq
        fault_params[idx] = (amplitude, frequency, phase)

    print("\n=== Randomly Generated Time‑Varying Faults ===")
    for idx in fault_indices:
        amp, freq, ph = fault_params[idx]
        print(f"Anchor {idx}: amplitude={amp:+.2f}m, frequency={freq:.2f} cycles/duration")

    # MDB comparison: use the maximum absolute bias as a rough reference
    max_abs_bias = {idx: np.max(np.abs(seq)) for idx, seq in fault_bias_sequences.items()}
    print("\n=== MDB & Fault Detectability (based on max |bias|) ===")
    for idx in fault_indices:
        print(f"Anchor {idx}: max|bias|={max_abs_bias[idx]:.3f}m, MDB={mdb_values[idx]:.4f}m -> "
              f"{'Detectable' if max_abs_bias[idx] > mdb_values[idx] else 'Not detectable'}")

    # ---- Storage ----
    est_traj = np.zeros((num_steps, 2))
    corrected_traj = np.zeros((num_steps, 2))
    wsse_vals = np.zeros(num_steps)
    hpl_vals = np.zeros(num_steps)
    alarm_fde = np.zeros(num_steps, dtype=bool)
    alarm_araim = np.zeros(num_steps, dtype=bool)
    excluded_history = []
    fault_active = np.zeros(num_steps, dtype=bool)

    # ---- Simulation loop ----
    for k in range(num_steps):
        true_pos = true_traj[k]
        true_ranges = compute_ranges(true_pos, anchors)
        noise = np.random.normal(0, sigma, N_anchors)
        measurements = true_ranges + noise

        # Inject time‑varying biases if within fault interval
        if fault_start <= k < fault_end:
            for idx in fault_indices:
                measurements[idx] += fault_bias_sequences[idx][k]
            fault_active[k] = True

        # ---- 1. WLS without FDE (baseline) ----
        est_traj[k] = wls_position(measurements, anchors, W_full)

        # ---- 2. Iterative FDE ----
        pos_corrected, excl_list, _ = wls_position_with_iterative_fde(
            measurements, anchors, W_full, alpha
        )
        corrected_traj[k] = pos_corrected
        excluded_history.append(excl_list)
        alarm_fde[k] = (len(excl_list) > 0)

        # ---- 3. ARAIM Protection Level ----
        hpl, alarm, _ = compute_araim_pl(measurements, anchors, W_full, alpha_pl, AL)
        hpl_vals[k] = hpl
        alarm_araim[k] = alarm

        # WSSE (for diagnostic plots)
        wsse = compute_wsse(measurements, est_traj[k], anchors, W_full)
        wsse_vals[k] = wsse

    # ---- Post‑processing: errors ----
    error_no = np.linalg.norm(est_traj - true_traj, axis=1)
    error_fde = np.linalg.norm(corrected_traj - true_traj, axis=1)

    # ================================
    # 8. Plotting
    # ================================
    # ---- Trajectory ----
    plt.figure(figsize=(12, 8))
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'k-', lw=2, label='True')
    plt.plot(est_traj[:, 0], est_traj[:, 1], 'b--', lw=1.5, label='WLS w/o FDE')
    plt.plot(corrected_traj[:, 0], corrected_traj[:, 1], 'g-', lw=2, label='WLS with FDE')
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
    plt.title('Trajectory with Time‑Varying Faults and FDE')
    plt.grid(True); plt.axis('equal'); plt.legend()
    plt.tight_layout()

    # ---- Error and HPL vs time ----
    plt.figure(figsize=(12, 4))
    plt.plot(error_no, 'b-', label='Error w/o FDE')
    plt.plot(error_fde, 'g-', label='Error with FDE')
    plt.plot(hpl_vals, 'm-', lw=1.5, label='ARAIM HPL')
    plt.axvspan(fault_start, fault_end-1, alpha=0.2, color='red', label='Faults')
    plt.xlabel('Time step'); plt.ylabel('Position error (m)')
    plt.title('Position Error and ARAIM HPL')
    plt.legend(); plt.grid(True); plt.tight_layout()

    # ---- HPL alone ----
    plt.figure(figsize=(12, 4))
    plt.plot(hpl_vals, 'm-', lw=1.5, label='ARAIM HPL')
    plt.axhline(y=AL, color='r', linestyle='--', label=f'Alert Limit = {AL} m')
    plt.axvspan(fault_start, fault_end-1, alpha=0.2, color='red')
    plt.xlabel('Time step'); plt.ylabel('HPL (m)')
    plt.title('ARAIM Horizontal Protection Level')
    plt.legend(); plt.grid(True); plt.tight_layout()

    # ---- WSSE and exclusions ----
    plt.figure(figsize=(12, 4))
    plt.plot(wsse_vals, 'b-', label='WSSE')
    threshold_all = chi2.ppf(1-alpha, N_anchors-state_dim)
    plt.axhline(y=threshold_all, color='r', linestyle='--', label=f'Threshold (all {N_anchors})')
    plt.axvspan(fault_start, fault_end-1, alpha=0.2, color='red')
    for k, excl in enumerate(excluded_history):
        if excl:
            plt.text(k, wsse_vals[k]+0.5, ','.join(map(str, excl)),
                     fontsize=7, color='purple', alpha=0.7)
    plt.xlabel('Time step'); plt.ylabel('WSSE')
    plt.title('Test Statistic and Excluded Anchors (iterative FDE)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # ================================
    # 9. Statistics
    # ================================
    total_fde_alarms = np.sum(alarm_fde)
    true_fault_steps = np.sum(fault_active)
    detected = np.sum(alarm_fde & fault_active)
    false_alarms = np.sum(alarm_fde & ~fault_active)

    print("\n=== Performance Summary ===")
    print(f"Faulty anchors: {fault_indices}")
    print(f"Fault duration: steps {fault_start} to {fault_end-1}")
    print(f"Steps with faults: {true_fault_steps}")
    print(f"FDE alarms raised: {total_fde_alarms}")
    print(f"Correct detections: {detected}")
    print(f"False alarms: {false_alarms}")
    if true_fault_steps > 0:
        print(f"FDE detection rate: {detected/true_fault_steps:.3f}")
    print(f"Mean error w/o FDE: {np.mean(error_no):.3f} m")
    print(f"Mean error with FDE: {np.mean(error_fde):.3f} m")
    improvement = (np.mean(error_no)-np.mean(error_fde))/np.mean(error_no)*100 if np.mean(error_no)>0 else 0
    print(f"Improvement: {improvement:.1f}%")

    fault_excl = [excl for k, excl in enumerate(excluded_history) if fault_active[k]]
    if fault_excl:
        print(f"Avg excluded anchors during faults: {np.mean([len(e) for e in fault_excl]):.2f}")

    total_araim_alarms = np.sum(alarm_araim)
    false_araim = np.sum(alarm_araim & ~fault_active)
    missed_araim = np.sum(~alarm_araim & fault_active)
    print(f"\nARAIM (HPL > {AL} m):")
    print(f"  Alarms raised: {total_araim_alarms}")
    print(f"  False alarms: {false_araim}")
    print(f"  Missed detections: {missed_araim}")
    if true_fault_steps > 0:
        print(f"  Detection rate: {1 - missed_araim/true_fault_steps:.3f}")
    print(f"  Mean HPL: {np.mean(hpl_vals):.3f} m, Max HPL: {np.max(hpl_vals):.3f} m")


# ------------------------------
# 10. Run
# ------------------------------
if __name__ == "__main__":
    run_trajectory_simulation()