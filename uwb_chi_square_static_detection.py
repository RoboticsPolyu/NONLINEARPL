"""
UWB Positioning Simulation with Chi‑Square Fault Detection
===========================================================
This script simulates a 2D UWB localization system with multiple anchors.
It generates range measurements with Gaussian noise and optionally injects
biases (faults) into one or more measurements. The weighted least‑squares
(WLS) solution is computed, and the weighted sum of squared errors (WSSE)
is used as the test statistic. Under nominal conditions, WSSE follows a
chi‑square distribution. We compare the statistic against a threshold to
detect faults and evaluate the detection performance via Monte Carlo runs.
"""

import numpy as np
from scipy.stats import chi2
from scipy.linalg import inv
import matplotlib.pyplot as plt

# ------------------------------
# 1. Simulation Parameters
# ------------------------------
np.random.seed(42)          # for reproducibility

# Anchor positions (2D) [x, y] in meters
anchors = np.array([
    [0.0, 0.0],
    [5.0, 0.0],
    [0.0, 10.0],
    [5.0, 10.0]
])
N_anchors = anchors.shape[0]

# True tag position
true_pos = np.array([5.0, 5.0])

# Measurement noise standard deviation (meters)
sigma = 0.1
# Precision matrix (diagonal)
W = np.diag(1.0 / sigma**2 * np.ones(N_anchors))

# Chi‑square threshold significance level (e.g., 0.05)
alpha = 0.05
# Degrees of freedom = N_anchors - state_dim (2 for 2D)
dof = N_anchors - 2
threshold = chi2.ppf(1 - alpha, dof)

# Fault parameters
fault_magnitude_range = np.linspace(0.0, 1.0, 21)  # bias added (meters)
num_runs = 500                # Monte Carlo runs per fault magnitude

# ------------------------------
# 2. Helper Functions
# ------------------------------
def compute_ranges(pos, anchors):
    """Compute Euclidean distances from a position to all anchors."""
    return np.sqrt(np.sum((anchors - pos)**2, axis=1))

def wls_position(measurements, anchors, W):
    """
    Weighted Least‑Squares position estimate.
    Uses Gauss‑Newton with a fixed number of iterations.
    """
    # Initial guess: centroid of anchors
    x = np.mean(anchors, axis=0)
    for _ in range(10):
        r = compute_ranges(x, anchors)
        residuals = measurements - r
        # Jacobian: each row = (x - a_i) / ||x - a_i||
        H = np.zeros((N_anchors, 2))
        for i in range(N_anchors):
            diff = x - anchors[i]
            norm = np.linalg.norm(diff)
            if norm > 1e-6:
                H[i, :] = diff / norm
            else:
                H[i, :] = np.array([0.0, 0.0])
        # Weighted normal equation
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
    """
    Compute the Weighted Sum of Squared Errors (WSSE) test statistic.
    WSSE = (z - h(x))^T W (z - h(x))
    """
    r_est = compute_ranges(pos_est, anchors)
    e = measurements - r_est
    return e.T @ W @ e

# ------------------------------
# 3. Monte Carlo Simulation
# ------------------------------
detection_rate = []
false_alarm_rate = []

print("Fault magnitude (m) | Detection Rate | False Alarm Rate")
print("---------------------------------------------------------")

for fmag in fault_magnitude_range:
    detections = 0
    false_alarms = 0
    for run in range(num_runs):
        # Generate nominal measurements with Gaussian noise
        true_ranges = compute_ranges(true_pos, anchors)
        noise = np.random.normal(0, sigma, N_anchors)
        measurements = true_ranges + noise

        # Inject fault (bias) into the first anchor if magnitude > 0
        if fmag > 0:
            measurements[0] += fmag   # add constant bias to first anchor

        # Estimate position
        pos_est = wls_position(measurements, anchors, W)
        # Compute test statistic
        wsse = compute_wsse(measurements, pos_est, anchors, W)

        # Decision: reject if WSSE > threshold
        is_fault = (fmag > 0)
        is_alarm = (wsse > threshold)

        if is_fault and is_alarm:
            detections += 1
        if not is_fault and is_alarm:
            false_alarms += 1

    # Rates for this fault magnitude
    det_rate = detections / num_runs if fmag > 0 else np.nan
    fa_rate = false_alarms / num_runs
    detection_rate.append(det_rate)
    false_alarm_rate.append(fa_rate)
    if fmag > 0:
        print(f"{fmag:6.3f}             | {det_rate:8.3f}        | {fa_rate:8.3f}")
    else:
        print(f"{fmag:6.3f}             |      N/A         | {fa_rate:8.3f}")

# ------------------------------
# 4. Plotting Results
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(fault_magnitude_range[1:], detection_rate[1:], 'b-o', label='Detection Rate')
plt.axhline(y=alpha, color='r', linestyle='--', label=f'Nominal False Alarm Rate ({alpha:.2f})')
plt.xlabel('Fault Magnitude (bias added to first anchor, m)')
plt.ylabel('Rate')
plt.title('Chi‑Square Fault Detection Performance in UWB Positioning')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print the achieved false alarm rate at zero fault (should be close to alpha)
print(f"\nAchieved false alarm rate at zero fault: {false_alarm_rate[0]:.3f} (expected {alpha:.3f})")