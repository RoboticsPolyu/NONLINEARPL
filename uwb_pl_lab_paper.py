import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools
from scipy.stats import chi2

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_anchors = 4  # UWB anchors
true_position = np.array([5.0, 6.0])  # True tag position [x, y]
anchor_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])  # Anchor positions

# Noise parameters
gaussian_std = 0.1  # Standard deviation of Gaussian noise
bias_magnitude = 1.5  # Magnitude of bias in one anchor
biased_anchor_idx = 2  # Index of the anchor with bias

# Generate measurements
true_distances = np.linalg.norm(anchor_positions - true_position, axis=1)
noisy_distances = true_distances + np.random.normal(0, gaussian_std, num_anchors)
noisy_distances[biased_anchor_idx] += bias_magnitude  # Add bias to one anchor

# Weight matrix (inverse of covariance)
W = np.eye(num_anchors) / (gaussian_std ** 2)

# Observation function (nonlinear)
def h(x):
    return np.linalg.norm(anchor_positions - x, axis=1)

# Jacobian of h
def jacobian_h(x):
    diffs = x - anchor_positions
    distances = np.linalg.norm(diffs, axis=1, keepdims=True)
    # Avoid division by zero
    distances[distances < 1e-6] = 1e-6
    return diffs / distances

# Initial estimate (least squares solution)
def least_squares_estimate():
    def cost(x):
        return np.sum((noisy_distances - h(x))**2)
    
    res = minimize(cost, [0, 0], method='BFGS')
    return res.x if res.success else np.array([5.0, 5.0])

x0 = least_squares_estimate()

# Linearized model around x0
J = jacobian_h(x0)
y = noisy_distances - h(x0)

# Calculate S matrix (Equation 20 in paper)
JWJ_inv = np.linalg.inv(J.T @ W @ J)
S = W @ (np.eye(num_anchors) - J @ JWJ_inv @ J.T @ W)

# Chi-square threshold (95% confidence, m-n degrees of freedom)
alpha = 0.05
chi2_threshold = chi2.ppf(1 - alpha, num_anchors - 2)

# Calculate 3-sigma protection level for Gaussian noise only
sigma_max = np.sqrt(np.max(np.linalg.eigvals(JWJ_inv)))
three_sigma_pl = 3 * sigma_max

# NEW: Paper-based protection level calculation with multiple fault assumption
def calculate_protection_level_paper_method(max_outliers=2):
    """
    Calculate protection level based on the paper's method considering multiple outliers
    """
    n = num_anchors  # number of measurements
    m = 2  # number of state variables (x, y)
    
    # Direction vectors for position (x, y)
    # For 2D position, we use 2x1 vectors
    H_x = np.array([[1, 0]])  # x-direction
    H_y = np.array([[0, 1]])  # y-direction
    
    # Calculate D matrices for each direction (Equation 27 in paper)
    # D_i = W J (J^T W J)^{-1} H_i^T H_i (J^T W J)^{-1} J^T W
    JWJ_inv = np.linalg.inv(J.T @ W @ J)
    
    # For x-direction
    term_x = JWJ_inv @ H_x.T @ H_x @ JWJ_inv
    D_x = W @ J @ term_x @ J.T @ W
    
    # For y-direction  
    term_y = JWJ_inv @ H_y.T @ H_y @ JWJ_inv
    D_y = W @ J @ term_y @ J.T @ W
    
    max_pl_bias_x = 0
    max_pl_bias_y = 0
    
    # Iterate over all possible outlier combinations
    for r in range(1, min(max_outliers + 1, n - m + 1)):
        # Generate all combinations of r outliers
        outlier_combinations = list(itertools.combinations(range(n), r))
        
        for outlier_indices in outlier_combinations:
            # Create selection matrix A_j (n x r)
            A_j = np.zeros((n, r))
            for idx, outlier_idx in enumerate(outlier_indices):
                A_j[outlier_idx, idx] = 1
            
            # Calculate the matrices for this outlier combination
            ATSA = A_j.T @ S @ A_j  # A_j^T S A_j
            ATDxA = A_j.T @ D_x @ A_j  # A_j^T D_x A_j
            ATDyA = A_j.T @ D_y @ A_j  # A_j^T D_y A_j
            
            # Ensure matrices are invertible and calculate eigenvalues
            if np.linalg.matrix_rank(ATSA) == r and np.linalg.cond(ATSA) < 1e10:
                try:
                    # Calculate eigenvalues for x-direction (Equation 32)
                    eigvals_x = np.linalg.eigvals(ATDxA @ np.linalg.inv(ATSA))
                    max_eig_x = np.max(np.real(eigvals_x))
                    
                    # Calculate eigenvalues for y-direction
                    eigvals_y = np.linalg.eigvals(ATDyA @ np.linalg.inv(ATSA))
                    max_eig_y = np.max(np.real(eigvals_y))
                    
                    # Calculate bias-induced protection level (Equation 31)
                    pl_bias_x = np.sqrt(max_eig_x * chi2_threshold)
                    pl_bias_y = np.sqrt(max_eig_y * chi2_threshold)
                    
                    if pl_bias_x > max_pl_bias_x:
                        max_pl_bias_x = pl_bias_x
                    if pl_bias_y > max_pl_bias_y:
                        max_pl_bias_y = pl_bias_y
                        
                except np.linalg.LinAlgError:
                    continue
    
    # Calculate noise-induced protection level (3-sigma rule)
    pl_noise_x = 3 * np.sqrt(JWJ_inv[0, 0])  # x-direction
    pl_noise_y = 3 * np.sqrt(JWJ_inv[1, 1])  # y-direction
    
    # Total protection level (bias + noise) - Equation 24
    total_pl_x = max_pl_bias_x + pl_noise_x
    total_pl_y = max_pl_bias_y + pl_noise_y
    
    # Overall protection level (conservative bound using Euclidean norm)
    total_pl = np.sqrt(total_pl_x**2 + total_pl_y**2)
    
    return total_pl, total_pl_x, total_pl_y, max_pl_bias_x, max_pl_bias_y, pl_noise_x, pl_noise_y

# Calculate protection level using paper method
pl_paper, pl_x, pl_y, bias_pl_x, bias_pl_y, noise_pl_x, noise_pl_y = calculate_protection_level_paper_method(max_outliers=1)

# Simple bias-induced PL calculation for comparison
def calculate_simple_bias_pl():
    """Simple calculation for comparison"""
    JWJ_inv = np.linalg.inv(J.T @ W @ J)
    bias_vec = np.zeros(num_anchors)
    bias_vec[biased_anchor_idx] = bias_magnitude
    
    dx = JWJ_inv @ J.T @ W @ bias_vec
    return np.linalg.norm(dx)

simple_bias_pl = calculate_simple_bias_pl()

# Visualization
plt.figure(figsize=(12, 10))

# Plot anchors
plt.scatter(anchor_positions[:, 0], anchor_positions[:, 1], s=100, c='blue', marker='^', label='UWB Anchors')
for i, pos in enumerate(anchor_positions):
    plt.annotate(f'A{i}', (pos[0] + 0.2, pos[1] + 0.2), fontsize=12)

# Highlight biased anchor
plt.scatter(anchor_positions[biased_anchor_idx, 0], 
            anchor_positions[biased_anchor_idx, 1], 
            s=200, facecolors='none', edgecolors='red', linewidths=2, label='Biased Anchor')

# Plot true position
plt.scatter(true_position[0], true_position[1], s=100, c='green', marker='o', label='True Position')

# Plot estimated position
plt.scatter(x0[0], x0[1], s=100, c='orange', marker='x', label='LS Estimate')

# Plot 3-sigma circle for Gaussian noise
circle_3sigma = plt.Circle(x0, three_sigma_pl, fill=False, color='purple', linestyle=':', 
                           linewidth=2, label='3σ Gaussian Uncertainty')
plt.gca().add_patch(circle_3sigma)

# Plot simple bias-induced protection level circle
simple_bias_circle = plt.Circle(x0, simple_bias_pl + three_sigma_pl, fill=False, color='blue', linestyle='-.', 
                               linewidth=2, label='Simple Bias PL + 3σ')
plt.gca().add_patch(simple_bias_circle)

# Plot protection level circle (paper method)
circle_pl_paper = plt.Circle(x0, pl_paper, fill=False, color='red', linestyle='-', 
                       linewidth=3, label='Paper Method PL')
plt.gca().add_patch(circle_pl_paper)

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title(f'UWB Protection Level Simulation\nPaper PL = {pl_paper:.2f}m, Simple Bias PL = {simple_bias_pl:.2f}m, 3σ = {three_sigma_pl:.2f}m\nTrue Error = {np.linalg.norm(x0 - true_position):.2f}m')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.axis('equal')
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.tight_layout()
plt.show()

# Print detailed results
print("=" * 60)
print("UWB PROTECTION LEVEL ANALYSIS")
print("=" * 60)
print(f"True position: {true_position}")
print(f"Estimated position: {x0}")
print(f"Position error: {np.linalg.norm(x0 - true_position):.3f} m")
print(f"3-sigma Protection Level (Gaussian only): {three_sigma_pl:.3f} m")
print(f"Simple Bias-induced PL: {simple_bias_pl:.3f} m")
print(f"Paper Method Protection Level: {pl_paper:.3f} m")
print("\nDirectional Analysis:")
print(f"  X-direction: PL_x = {pl_x:.3f} m")
print(f"    - Bias component: {bias_pl_x:.3f} m") 
print(f"    - Noise component: {noise_pl_x:.3f} m")
print(f"  Y-direction: PL_y = {pl_y:.3f} m")
print(f"    - Bias component: {bias_pl_y:.3f} m")
print(f"    - Noise component: {noise_pl_y:.3f} m")
print(f"\nTrue bias: Anchor {biased_anchor_idx} with magnitude {bias_magnitude} m")
print("=" * 60)