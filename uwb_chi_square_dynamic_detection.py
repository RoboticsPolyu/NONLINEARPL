"""
UWB Positioning Simulation with Chi‑Square Fault Detection
===========================================================
This script provides two complementary analyses:
1. Monte Carlo evaluation of detection performance vs. fault magnitude.
2. A trajectory simulation that plots the estimated path, positioning errors,
   and marks fault detection events.

The same UWB setup (4 anchors, Gaussian noise, WLS estimation) is used.
"""

import numpy as np
from scipy.stats import chi2
from scipy.linalg import inv
import matplotlib.pyplot as plt

# ------------------------------
# Common Parameters
# ------------------------------
np.random.seed(42)

# Anchor positions (2D)
anchors = np.array([
    [0.0, 0.0],
    [5.0, 0.0],
    [0.0, 10.0],
    [5.0, 10.0],
    [10.0, 5.0],
    [10.0, 10.0]
])
N_anchors = anchors.shape[0]

# Measurement noise standard deviation (m)
sigma = 0.1
W = np.diag(1.0 / sigma**2 * np.ones(N_anchors))

# Chi‑square threshold (α = 0.05)
alpha = 0.05
dof = N_anchors - 2          # 2D position
threshold = chi2.ppf(1 - alpha, dof)

# ------------------------------
# Helper Functions (same as before)
# ------------------------------
def compute_ranges(pos, anchors):
    """Euclidean distances from a position to all anchors."""
    return np.sqrt(np.sum((anchors - pos)**2, axis=1))

def wls_position(measurements, anchors, W):
    """Weighted Least‑Squares position estimate via Gauss‑Newton."""
    x = np.mean(anchors, axis=0)          # initial guess
    for _ in range(10):
        r = compute_ranges(x, anchors)
        residuals = measurements - r
        H = np.zeros((N_anchors, 2))
        for i in range(N_anchors):
            diff = x - anchors[i]
            norm = np.linalg.norm(diff)
            if norm > 1e-6:
                H[i, :] = diff / norm
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
    """Weighted Sum of Squared Errors (test statistic)."""
    r_est = compute_ranges(pos_est, anchors)
    e = measurements - r_est
    return e.T @ W @ e

# -------------------------------------------------------------------
# 1. Monte Carlo Performance Evaluation (optional)
# -------------------------------------------------------------------
def run_monte_carlo(true_pos=np.array([5.0, 5.0])):
    """
    Evaluate detection rate vs. fault magnitude.
    true_pos: true tag position for the static simulation.
    """
    fault_magnitude_range = np.linspace(0.0, 1.0, 21)
    num_runs = 500
    detection_rate = []
    false_alarm_rate = []

    print("Fault magnitude (m) | Detection Rate | False Alarm Rate")
    print("---------------------------------------------------------")

    for fmag in fault_magnitude_range:
        detections = 0
        false_alarms = 0
        for run in range(num_runs):
            true_ranges = compute_ranges(true_pos, anchors)
            noise = np.random.normal(0, sigma, N_anchors)
            measurements = true_ranges + noise
            if fmag > 0:
                measurements[0] += fmag       # bias on first anchor
            pos_est = wls_position(measurements, anchors, W)
            wsse = compute_wsse(measurements, pos_est, anchors, W)
            is_fault = (fmag > 0)
            is_alarm = (wsse > threshold)
            if is_fault and is_alarm:
                detections += 1
            if not is_fault and is_alarm:
                false_alarms += 1

        det_rate = detections / num_runs if fmag > 0 else np.nan
        fa_rate = false_alarms / num_runs
        detection_rate.append(det_rate)
        false_alarm_rate.append(fa_rate)
        if fmag > 0:
            print(f"{fmag:6.3f}             | {det_rate:8.3f}        | {fa_rate:8.3f}")
        else:
            print(f"{fmag:6.3f}             |      N/A         | {fa_rate:8.3f}")

    # Plot detection rate curve
    plt.figure(figsize=(8, 5))
    plt.plot(fault_magnitude_range[1:], detection_rate[1:], 'b-o', label='Detection Rate')
    plt.axhline(y=alpha, color='r', linestyle='--', label=f'Nominal False Alarm Rate ({alpha:.2f})')
    plt.xlabel('Fault Magnitude (bias added to first anchor, m)')
    plt.ylabel('Rate')
    plt.title('Chi‑Square Fault Detection Performance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nAchieved false alarm rate at zero fault: {false_alarm_rate[0]:.3f} (expected {alpha:.3f})")

def run_trajectory_simulation():
    """
    Simulate a moving tag along a straight line, estimate its position,
    apply the chi‑square test at each epoch, and plot:
        - trajectory (true vs. estimated) with alarm markers
        - positioning errors (X and Y) over time with alarm markers
          and annotations showing the suspected faulty anchor ID
        - WSSE time series with threshold
    """
    # Generate a straight‑line trajectory from (1,1) to (9,9)
    num_steps = 500
    t = np.linspace(0, 1, num_steps)
    true_traj = np.column_stack((1 + 8*t, 1 + 8*t))   # x = y

    # Fault injection: first anchor gets a bias during a segment
    fault_start = 150
    fault_end = 400
    fault_magnitude = 0.5   # meters

    # Storage for estimates, test statistics, and suspected anchor IDs
    est_traj = np.zeros((num_steps, 2))
    wsse_vals = np.zeros(num_steps)
    alarm_flags = np.zeros(num_steps, dtype=bool)
    fault_active = np.zeros(num_steps, dtype=bool)
    suspected_anchor = -np.ones(num_steps, dtype=int)   # -1 means no alarm

    for k in range(num_steps):
        true_pos = true_traj[k]
        true_ranges = compute_ranges(true_pos, anchors)
        noise = np.random.normal(0, sigma, N_anchors)
        measurements = true_ranges + noise

        # Inject fault only during the specified interval
        if fault_start <= k < fault_end:
            measurements[0] += fault_magnitude
            fault_active[k] = True

        # Estimate position
        pos_est = wls_position(measurements, anchors, W)
        est_traj[k] = pos_est

        # Compute residuals and WSSE
        r_est = compute_ranges(pos_est, anchors)
        residuals = measurements - r_est
        wsse = residuals.T @ W @ residuals
        wsse_vals[k] = wsse

        # Test against threshold
        alarm = (wsse > threshold)
        alarm_flags[k] = alarm

        if alarm:
            # Find the anchor with the largest absolute residual
            # (standardised residual could also be used, but absolute is simpler)
            abs_res = np.abs(residuals)
            suspected_anchor[k] = np.argmax(abs_res)

    # Compute positioning errors
    errors = est_traj - true_traj

    # ------------------------------
    # Plot 1: Trajectory (true vs. estimated) with alarm markers
    # ------------------------------
    plt.figure(figsize=(10, 8))

    plt.plot(true_traj[:, 0], true_traj[:, 1], 'k-', linewidth=2, label='True trajectory')
    plt.plot(est_traj[:, 0], est_traj[:, 1], 'b--', linewidth=1.5, label='Estimated trajectory')

    alarm_indices = np.where(alarm_flags)[0]
    if len(alarm_indices) > 0:
        plt.scatter(est_traj[alarm_indices, 0], est_traj[alarm_indices, 1],
                    c='red', s=50, marker='x', label='Alarm (fault detected)')

    if fault_start < fault_end:
        fault_idx = np.where(fault_active)[0]
        if len(fault_idx) > 0:
            plt.plot(true_traj[fault_idx, 0], true_traj[fault_idx, 1],
                     'r-', linewidth=4, alpha=0.3, label='True fault period')

    plt.scatter(anchors[:, 0], anchors[:, 1], c='green', s=100, marker='^', label='Anchors')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('UWB Positioning Trajectory with Chi‑Square Fault Detection')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    # ------------------------------
    # Plot 2: Positioning errors over time with alarm markers and anchor ID annotations
    # ------------------------------
    plt.figure(figsize=(12, 7))

    # Subplot 2a: X error
    plt.subplot(2, 1, 1)
    plt.plot(errors[:, 0], 'b-', label='X error (est - true)')
    # Mark alarm points and annotate with suspected anchor ID
    for idx in alarm_indices:
        plt.scatter(idx, errors[idx, 0], c='red', s=40, marker='x')
        # Add text annotation above the marker
        plt.text(idx, errors[idx, 0] + 0.02, f'{suspected_anchor[idx]}',
                 fontsize=8, color='red', ha='center', va='bottom')
    plt.axvspan(fault_start, fault_end-1, alpha=0.2, color='red', label='Fault active')
    plt.ylabel('X error (m)')
    plt.title('X‑Error with Fault Detection and Suspected Anchor ID')
    plt.grid(True)
    plt.legend(['X error', 'Alarm', 'Fault active'])

    # Subplot 2b: Y error
    plt.subplot(2, 1, 2)
    plt.plot(errors[:, 1], 'g-', label='Y error (est - true)')
    for idx in alarm_indices:
        plt.scatter(idx, errors[idx, 1], c='red', s=40, marker='x')
        plt.text(idx, errors[idx, 1] + 0.02, f'{suspected_anchor[idx]}',
                 fontsize=8, color='red', ha='center', va='bottom')
    plt.axvspan(fault_start, fault_end-1, alpha=0.2, color='red', label='Fault active')
    plt.xlabel('Time step')
    plt.ylabel('Y error (m)')
    plt.title('Y‑Error with Fault Detection and Suspected Anchor ID')
    plt.grid(True)
    plt.legend(['Y error', 'Alarm', 'Fault active'])
    plt.tight_layout()

    # ------------------------------
    # Plot 3: WSSE time series with threshold
    # ------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(wsse_vals, 'b-', label='WSSE statistic')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (α={alpha:.2f})')
    plt.axvspan(fault_start, fault_end-1, alpha=0.2, color='red', label='Fault active')
    plt.xlabel('Time step')
    plt.ylabel('WSSE')
    plt.title('Test Statistic Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print summary
    total_alarms = np.sum(alarm_flags)
    true_fault_steps = np.sum(fault_active)
    detected_fault_steps = np.sum(alarm_flags & fault_active)
    false_alarms = np.sum(alarm_flags & ~fault_active)
    print(f"\nTrajectory simulation summary:")
    print(f"  Total steps: {num_steps}")
    print(f"  Steps with fault: {true_fault_steps}")
    print(f"  Steps with alarm: {total_alarms}")
    print(f"  Correct detections (fault & alarm): {detected_fault_steps}")
    print(f"  False alarms (no fault but alarm): {false_alarms}")
    if true_fault_steps > 0:
        print(f"  Detection rate during fault period: {detected_fault_steps/true_fault_steps:.3f}")
    # Print suspected anchor distribution
    if total_alarms > 0:
        unique, counts = np.unique(suspected_anchor[alarm_flags], return_counts=True)
        print("  Suspected anchor IDs (alarm steps):")
        for uid, cnt in zip(unique, counts):
            print(f"    Anchor {uid}: {cnt} times")

# -------------------------------------------------------------------
# Main: choose which analysis to run
# -------------------------------------------------------------------
if __name__ == "__main__":
    # To run the Monte Carlo evaluation instead, uncomment the line below:
    # run_monte_carlo(true_pos=np.array([5.0, 5.0]))

    # Default: run the trajectory simulation and plot
    run_trajectory_simulation()