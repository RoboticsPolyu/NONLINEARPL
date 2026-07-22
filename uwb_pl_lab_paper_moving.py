import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools
from scipy.stats import chi2

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_anchors = 4
anchor_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])  # Anchor positions

# Circular path parameters
center = np.array([5.0, 5.0])  # Center of the circle
radius = 3.0  # Radius of the circle
num_points = 36  # Number of points along the circle (10° increments)
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

# Generate circular path
true_positions = np.array([
    center + radius * np.array([np.cos(angle), np.sin(angle)])
    for angle in angles
])

# Noise parameters
gaussian_std = 0.1
bias_magnitude = 1.5
biased_anchor_idx = 2

# Observation function
def h(x):
    return np.linalg.norm(anchor_positions - x, axis=1)

# Jacobian of h
def jacobian_h(x):
    diffs = x - anchor_positions
    distances = np.linalg.norm(diffs, axis=1, keepdims=True)
    distances[distances < 1e-6] = 1e-6
    return diffs / distances

# Least squares estimation
def least_squares_estimate(measurements):
    def cost(x):
        return np.sum((measurements - h(x))**2)
    
    res = minimize(cost, [5.0, 5.0], method='BFGS')
    return res.x if res.success else np.array([5.0, 5.0])

# Paper-based protection level calculation
def calculate_protection_level_paper_method(J, W, max_outliers=2):
    n = num_anchors
    m = 2
    
    JWJ_inv = np.linalg.inv(J.T @ W @ J)
    S = W @ (np.eye(n) - J @ JWJ_inv @ J.T @ W)
    
    # Chi-square threshold
    alpha = 0.05
    chi2_threshold = chi2.ppf(1 - alpha, n - m)
    
    # Direction vectors
    H_x = np.array([[1, 0]])
    H_y = np.array([[0, 1]])
    
    # D matrices
    term_x = JWJ_inv @ H_x.T @ H_x @ JWJ_inv
    D_x = W @ J @ term_x @ J.T @ W
    
    term_y = JWJ_inv @ H_y.T @ H_y @ JWJ_inv
    D_y = W @ J @ term_y @ J.T @ W
    
    max_pl_bias_x = 0
    max_pl_bias_y = 0
    
    # Iterate over outlier combinations
    for r in range(1, min(max_outliers + 1, n - m + 1)):
        outlier_combinations = list(itertools.combinations(range(n), r))
        
        for outlier_indices in outlier_combinations:
            A_j = np.zeros((n, r))
            for idx, outlier_idx in enumerate(outlier_indices):  # FIXED: Changed outoutlier_indices to outlier_indices
                A_j[outlier_idx, idx] = 1
            
            ATSA = A_j.T @ S @ A_j
            ATDxA = A_j.T @ D_x @ A_j
            ATDyA = A_j.T @ D_y @ A_j
            
            if np.linalg.matrix_rank(ATSA) == r and np.linalg.cond(ATSA) < 1e10:
                try:
                    eigvals_x = np.linalg.eigvals(ATDxA @ np.linalg.inv(ATSA))
                    eigvals_y = np.linalg.eigvals(ATDyA @ np.linalg.inv(ATSA))
                    
                    max_eig_x = np.max(np.real(eigvals_x))
                    max_eig_y = np.max(np.real(eigvals_y))
                    
                    pl_bias_x = np.sqrt(max_eig_x * chi2_threshold)
                    pl_bias_y = np.sqrt(max_eig_y * chi2_threshold)
                    
                    if pl_bias_x > max_pl_bias_x:
                        max_pl_bias_x = pl_bias_x
                    if pl_bias_y > max_pl_bias_y:
                        max_pl_bias_y = pl_bias_y
                        
                except np.linalg.LinAlgError:
                    continue
    
    # Noise-induced protection level
    pl_noise_x = 3 * np.sqrt(JWJ_inv[0, 0])
    pl_noise_y = 3 * np.sqrt(JWJ_inv[1, 1])
    
    # Total protection level
    total_pl_x = max_pl_bias_x + pl_noise_x
    total_pl_y = max_pl_bias_y + pl_noise_y
    total_pl = np.sqrt(total_pl_x**2 + total_pl_y**2)
    
    return total_pl, total_pl_x, total_pl_y

# Store results
estimated_positions = []
protection_levels = []
position_errors = []
angles_deg = np.degrees(angles)

print("Simulating robot movement along circular path...")
for i, true_pos in enumerate(true_positions):
    # Generate measurements with noise and bias
    true_distances = np.linalg.norm(anchor_positions - true_pos, axis=1)
    noisy_distances = true_distances + np.random.normal(0, gaussian_std, num_anchors)
    noisy_distances[biased_anchor_idx] += bias_magnitude
    
    # Estimate position
    W = np.eye(num_anchors) / (gaussian_std ** 2)
    estimated_pos = least_squares_estimate(noisy_distances)
    estimated_positions.append(estimated_pos)
    
    # Calculate protection level
    J = jacobian_h(estimated_pos)
    pl, pl_x, pl_y = calculate_protection_level_paper_method(J, W, max_outliers=1)
    protection_levels.append(pl)
    
    # Calculate position error
    error = np.linalg.norm(estimated_pos - true_pos)
    position_errors.append(error)
    
    if i % 6 == 0:  # Print every 60 degrees
        print(f"Angle {angles_deg[i]:.0f}°: True={true_pos}, Est={estimated_pos}, Error={error:.3f}m, PL={pl:.3f}m")

# Convert to numpy arrays for easier handling
protection_levels = np.array(protection_levels)
position_errors = np.array(position_errors)

# Create safe/unsafe masks
safe_mask = protection_levels >= position_errors
unsafe_mask = protection_levels < position_errors

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Trajectory and protection levels
ax1.scatter(anchor_positions[:, 0], anchor_positions[:, 1], s=100, c='blue', marker='^', label='UWB Anchors')
for i, pos in enumerate(anchor_positions):
    ax1.annotate(f'A{i}', (pos[0] + 0.2, pos[1] + 0.2), fontsize=10)

# Plot true trajectory
true_x, true_y = true_positions[:, 0], true_positions[:, 1]
ax1.plot(true_x, true_y, 'g-', linewidth=2, label='True Trajectory')
ax1.scatter(true_x, true_y, s=20, c='green', alpha=0.6)

# Plot estimated trajectory
est_x, est_y = np.array(estimated_positions)[:, 0], np.array(estimated_positions)[:, 1]
ax1.plot(est_x, est_y, 'r--', linewidth=1, label='Estimated Trajectory')
ax1.scatter(est_x, est_y, s=20, c='red', alpha=0.6)

# Color code points based on safety
safe_indices = np.where(safe_mask)[0]
unsafe_indices = np.where(unsafe_mask)[0]

if len(safe_indices) > 0:
    ax1.scatter(est_x[safe_indices], est_y[safe_indices], s=50, c='green', marker='o', label='Safe (PL ≥ Error)')
if len(unsafe_indices) > 0:
    ax1.scatter(est_x[unsafe_indices], est_y[unsafe_indices], s=50, c='red', marker='x', label='Unsafe (PL < Error)')

# Plot protection level circles at some key points
key_indices = [0, 9, 18, 27]  # 0°, 90°, 180°, 270°
for idx in key_indices:
    circle = plt.Circle(estimated_positions[idx], protection_levels[idx], 
                       fill=False, color='orange', linestyle='--', alpha=0.7,
                       linewidth=1)
    ax1.add_patch(circle)
    ax1.annotate(f'PL={protection_levels[idx]:.2f}m', 
                (estimated_positions[idx][0], estimated_positions[idx][1] + protection_levels[idx] + 0.3),
                fontsize=8, ha='center')

ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Robot Trajectory and Protection Levels')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')
ax1.set_xlim(-1, 11)
ax1.set_ylim(-1, 11)

# Plot 2: Protection Level vs Angle
ax2.plot(angles_deg, protection_levels, 'b-o', linewidth=2, markersize=4, label='Protection Level')
ax2.axhline(y=np.mean(protection_levels), color='r', linestyle='--', 
           label=f'Mean PL = {np.mean(protection_levels):.3f}m')

# Highlight safe and unsafe regions
ax2.fill_between(angles_deg, protection_levels, position_errors, 
                where=safe_mask, alpha=0.3, color='green', label='Safe Region')
ax2.fill_between(angles_deg, protection_levels, position_errors,
                where=unsafe_mask, alpha=0.3, color='red', label='Unsafe Region')

ax2.set_xlabel('Angle (degrees)')
ax2.set_ylabel('Protection Level (m)')
ax2.set_title('Protection Level Along Circular Path')
ax2.legend()
ax2.grid(True)
ax2.set_xticks(np.arange(0, 361, 45))

# Plot 3: Position Error vs Protection Level
ax3.plot(angles_deg, position_errors, 'r-o', linewidth=2, markersize=4, label='Position Error')
ax3.plot(angles_deg, protection_levels, 'b-o', linewidth=2, markersize=4, label='Protection Level')

# Create separate plots for safe and unsafe regions
safe_angles = angles_deg[safe_mask]
safe_errors = position_errors[safe_mask]
safe_pls = protection_levels[safe_mask]

unsafe_angles = angles_deg[unsafe_mask]
unsafe_errors = position_errors[unsafe_mask]
unsafe_pls = protection_levels[unsafe_mask]

if len(safe_angles) > 0:
    ax3.fill_between(safe_angles, safe_errors, safe_pls, alpha=0.3, color='green', label='Safe Region')
if len(unsafe_angles) > 0:
    ax3.fill_between(unsafe_angles, unsafe_errors, unsafe_pls, alpha=0.3, color='red', label='Unsafe Region')

ax3.set_xlabel('Angle (degrees)')
ax3.set_ylabel('Distance (m)')
ax3.set_title('Position Error vs Protection Level')
ax3.legend()
ax3.grid(True)
ax3.set_xticks(np.arange(0, 361, 45))

plt.tight_layout()
plt.show()

# Statistical analysis
safe_frames = np.sum(safe_mask)
safety_rate = safe_frames / num_points * 100

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)
print(f"Total frames: {num_points}")
print(f"Safe frames (PL ≥ Error): {safe_frames}")
print(f"Unsafe frames (PL < Error): {num_points - safe_frames}")
print(f"Safety rate: {safety_rate:.1f}%")
print(f"Mean Protection Level: {np.mean(protection_levels):.3f} ± {np.std(protection_levels):.3f} m")
print(f"Mean Position Error: {np.mean(position_errors):.3f} ± {np.std(position_errors):.3f} m")
print(f"Max Protection Level: {np.max(protection_levels):.3f} m at {angles_deg[np.argmax(protection_levels)]:.0f}°")
print(f"Max Position Error: {np.max(position_errors):.3f} m at {angles_deg[np.argmax(position_errors)]:.0f}°")

# Analyze protection level variation
pl_variation = np.max(protection_levels) - np.min(protection_levels)
print(f"Protection Level variation: {pl_variation:.3f} m")

# Additional safety analysis
max_error_exceedance = np.max(position_errors - protection_levels) if np.any(unsafe_mask) else 0
print(f"Maximum error exceedance over PL: {max_error_exceedance:.3f} m")
print("=" * 60)

# Print unsafe positions
if np.any(unsafe_mask):
    print("\nUNSAFE POSITIONS (PL < Error):")
    print("Angle(°) | Position Error | Protection Level | Difference")
    print("-" * 55)
    for idx in unsafe_indices:
        diff = position_errors[idx] - protection_levels[idx]
        print(f"{angles_deg[idx]:7.0f}° | {position_errors[idx]:13.3f}m | {protection_levels[idx]:15.3f}m | {diff:10.3f}m")