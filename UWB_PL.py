import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import cvxpy as cp
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

# Calculate S matrix
S = W @ (np.eye(num_anchors) - J @ np.linalg.inv(J.T @ W @ J) @ J.T @ W)

# Chi-square threshold (95% confidence, m-n degrees of freedom)
alpha = 0.05
chi2_threshold = chi2.ppf(1 - alpha, num_anchors - 2)

# Calculate 3-sigma protection level for Gaussian noise only
P = np.linalg.inv(J.T @ W @ J)
sigma_max = np.sqrt(np.max(np.linalg.eigvals(P)))
three_sigma_pl = 3 * sigma_max

# Calculate protection level using 𝛿𝐱 = (𝐉^𝑻 * 𝐖 * 𝐉)^-1 * 𝐉^𝑻 * 𝐖 * 𝐛
def calculate_bias_induced_pl():
    max_pl = 0
    max_bias_vec = None
    max_dx = None
    
    for i in range(num_anchors):
        bias_vec = np.zeros(num_anchors)
        bias_vec[i] = bias_magnitude
        
        dx = np.linalg.inv(J.T @ W @ J) @ J.T @ W @ bias_vec
        pl = np.linalg.norm(dx)
        
        if pl > max_pl:
            max_pl = pl
            max_bias_vec = bias_vec.copy()
            max_dx = dx.copy()
    
    return max_pl, max_bias_vec, max_dx

bias_induced_pl, max_bias_vec, max_dx = calculate_bias_induced_pl()

# NEW: Revised Mixed-integer optimization using linearized model
def calculate_protection_level_mio():
    # Define optimization variables
    dx1 = cp.Variable(2)  # Position error for first estimate
    dx2 = cp.Variable(2)  # Position error for second estimate
    lambda_vec = cp.Variable(num_anchors, boolean=True)  # Binary variables
    
    # Bias term (lambda * b)
    bias_term = bias_magnitude * lambda_vec
    
    # Linearized residuals (alpha_j = y - J @ dx_j)
    alpha1 = y - J @ dx1
    alpha2 = y - J @ dx2
    
    # Constraints
    constraints = []
    
    # Single bias constraint (only one anchor can have bias)
    constraints.append(cp.sum(lambda_vec) == 1)
    
    # Statistical test constraints from the paper
    # (alpha_j + lambda_j * b)^T S (alpha_j + lambda_j * b) <= chi2_threshold
    # constraints.append(cp.quad_form(alpha1 + bias_term, S) <= chi2_threshold)
    # constraints.append(cp.quad_form(alpha2 + bias_term, S) <= chi2_threshold)
    
    # Additional constraint: (lambda_j * b)^T S (lambda_j * b) <= Y*
    # For simplicity, we'll use chi2_threshold as Y*
    constraints.append(cp.quad_form(bias_term, S) <= chi2_threshold)
    
    # Measurement residual constraints (linearized version)
    # We approximate (z - h(x_j)) ≈ (y - J @ dx_j)
    constraints.append(cp.sum_squares(alpha1) <= chi2_threshold * 1.1)
    constraints.append(cp.sum_squares(alpha2) <= chi2_threshold * 1.1)
    
    # Since we cannot maximize norm directly, use epigraph trick
    t = cp.Variable(nonneg=True)
    constraints.append(cp.norm(dx1 - dx2) <= t)
    
    # Objective: maximize the distance between positions
    objective = cp.Maximize(t)
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    
    try:
        # Use ECOS_BB for mixed-integer optimization
        prob.solve(solver=cp.ECOS_BB, verbose=False, max_iters=1000)
        
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            protection_level = t.value
            x1_est = x0 + dx1.value
            x2_est = x0 + dx2.value
            return protection_level, x1_est, x2_est, lambda_vec.value
        else:
            print(f"MIO solver status: {prob.status}")
            raise Exception("MIO solver did not converge to optimal solution")
            
    except Exception as e:
        print(f"MIO solver error: {e}")
        # Fallback to exhaustive search
        return calculate_protection_level_exhaustive()

# Exhaustive search implementation
def calculate_protection_level_exhaustive():
    print("Using exhaustive search as fallback")
    max_pl = 0
    best_x1 = None
    best_x2 = None
    best_lambda = None
    
    inv_JWJ = np.linalg.inv(J.T @ W @ J)
    
    # Try all possible bias configurations
    for bias_idx in range(num_anchors):
        lambda_vec = np.zeros(num_anchors)
        lambda_vec[bias_idx] = 1
        bias_vec = bias_magnitude * lambda_vec
        
        # Check statistical constraints
        chi2_value_bias = bias_vec.T @ S @ bias_vec
        if chi2_value_bias > chi2_threshold:
            continue
            
        # Calculate position errors for two extreme cases
        dx_max = inv_JWJ @ J.T @ W @ bias_vec
        dx_min = -dx_max  # Opposite direction
        
        x1_candidate = x0 + dx_max
        x2_candidate = x0 + dx_min
        
        # Check measurement residuals (linearized)
        residual1 = y - J @ dx_max
        residual2 = y - J @ dx_min
        
        chi2_value1 = residual1.T @ S @ (residual1 + bias_vec)
        chi2_value2 = residual2.T @ S @ (residual2 + bias_vec)
        
        if chi2_value1 <= chi2_threshold and chi2_value2 <= chi2_threshold:
            pl_candidate = np.linalg.norm(x1_candidate - x2_candidate)
            if pl_candidate > max_pl:
                max_pl = pl_candidate
                best_x1 = x1_candidate
                best_x2 = x2_candidate
                best_lambda = lambda_vec
    
    if best_x1 is not None:
        return max_pl, best_x1, best_x2, best_lambda
    else:
        # Final fallback
        print("Exhaustive search failed, using geometric approach")
        fallback_lambda = np.zeros(num_anchors)
        fallback_lambda[biased_anchor_idx] = 1
        protection_level = 3 * sigma_max + bias_magnitude * np.max(np.abs(
            np.linalg.inv(J.T @ W @ J) @ J.T @ W @ fallback_lambda))
        return protection_level, x0, x0, fallback_lambda

# Calculate protection level
pl, x1_est, x2_est, lambda_est = calculate_protection_level_mio()

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
plt.scatter(true_position[0], true_position[1], s=15, c='green', marker='o', label='True Position')

# Plot estimated position
plt.scatter(x0[0], x0[1], s=5, c='orange', marker='x', label='LS Estimate')

# Plot 3-sigma circle for Gaussian noise
circle_3sigma = plt.Circle(x0, three_sigma_pl, fill=False, color='purple', linestyle=':', 
                           linewidth=1, label='3σ Gaussian Uncertainty')
plt.gca().add_patch(circle_3sigma)

# Plot bias-induced protection level circle
bias_pl_circle = plt.Circle(x0, bias_induced_pl + three_sigma_pl, fill=False, color='blue', linestyle='-.', 
                           linewidth=1, label='Bias-Induced PL + 3sigma')
plt.gca().add_patch(bias_pl_circle)

# Plot protection level circle (total uncertainty)
circle_pl = plt.Circle(x0, pl, fill=False, color='red', linestyle='--', 
                       linewidth=1, label='MIO Protection Level')
plt.gca().add_patch(circle_pl)

# Plot the position error due to bias
bias_induced_position = x0 + max_dx
plt.scatter(bias_induced_position[0], bias_induced_position[1], s=15, c='cyan', marker='s', 
           label='Max-dx Position')

# Plot the two extreme positions from the optimization if available
if np.linalg.norm(x1_est - x2_est) > 1e-6:
    plt.scatter(x1_est[0], x1_est[1], s=80, c='magenta', marker='+', label='Extreme Position 1')
    plt.scatter(x2_est[0], x2_est[1], s=80, c='brown', marker='+', label='Extreme Position 2')

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title(f'UWB Protection Level Simulation\nTotal PL = {pl:.2f}m, Bias PL = {bias_induced_pl:.2f}m, 3σ = {three_sigma_pl:.2f}m\nTrue Error = {np.linalg.norm(x0 - true_position):.2f}m')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.axis('equal')
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.tight_layout()
plt.show()

# Print results
print(f"True position: {true_position}")
print(f"Estimated position: {x0}")
print(f"Position error: {np.linalg.norm(x0 - true_position):.3f} m")
print(f"3-sigma Protection Level (Gaussian only): {three_sigma_pl:.3f} m")
print(f"Bias-induced Protection Level: {bias_induced_pl:.3f} m")
print(f"MIO Protection Level: {pl:.3f} m")
print(f"Lambda estimates from optimization: {lambda_est}")
print(f"Max bias vector from formula: {max_bias_vec}")
print(f"Position error from max bias: {max_dx}")
print(f"True bias was applied to anchor {biased_anchor_idx} with magnitude {bias_magnitude} m")