import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import least_squares
from scipy import stats
from scipy.linalg import inv

# Set random seed for reproducibility
np.random.seed(42)

# Define four anchor positions (x, y)
anchors = np.array([
    [0, 0],    # Anchor 1
    [10, 0],   # Anchor 2
    [10, 10],  # Anchor 3
    [0, 10]    # Anchor 4
])

# Define the true tag position
true_position = np.array([5.0, 6.0])

# Distance measurement function with noise and bias
def measure_distance(true_pos, anchor_pos, bias_anchor=None, bias_value=0):
    """Calculate distance measurement with noise and optional bias"""
    true_dist = np.linalg.norm(true_pos - anchor_pos)
    # Add Gaussian noise (standard deviation 5cm)
    noise = np.random.normal(0, 0.1)
    
    # Add bias to the specified anchor
    if bias_anchor is not None and np.array_equal(anchor_pos, bias_anchor):
        measured_dist = true_dist + noise + bias_value
    else:
        measured_dist = true_dist + noise
        
    return max(measured_dist, 0)  # Ensure non-negative distance

# Nonlinear least squares position estimation
def trilateration_error(x, anchors, measurements):
    """Error function for trilateration"""
    return [np.linalg.norm(x - anchors[i]) - measurements[i] for i in range(len(anchors))]

def estimate_position_with_outlier_rejection(anchors, measurements, max_iterations=10, alpha=0.05):
    """Estimate position using nonlinear least squares with outlier rejection"""
    # Initial guess (center of anchors)
    x0 = np.mean(anchors, axis=0)
    
    n_measurements = len(measurements)
    n_states = len(x0)
    
    # Initial covariance matrix for measurements (assuming equal variance)
    # In practice, this could be tuned based on sensor characteristics
    measurement_std = 0.1  # 10cm standard deviation
    Q = np.eye(n_measurements) * (measurement_std ** 2)
    W = inv(Q)  # Weight matrix
    
    # Start with all measurements included
    included_measurements = list(range(n_measurements))
    
    for iteration in range(max_iterations):
        # Create subset of anchors and measurements for current iteration
        current_anchors = anchors[included_measurements]
        current_measurements = [measurements[i] for i in included_measurements]
        current_W = W[np.ix_(included_measurements, included_measurements)]
        
        try:
            # Perform least squares estimation
            result = least_squares(trilateration_error, x0, args=(current_anchors, current_measurements))
            estimated_pos = result.x
            
            # Calculate residuals
            residuals = trilateration_error(estimated_pos, current_anchors, current_measurements)
            residuals = np.array(residuals)
            
            # Calculate Jacobian matrix numerically
            epsilon = 1e-8
            J = np.zeros((len(current_measurements), len(estimated_pos)))
            for i in range(len(estimated_pos)):
                x_plus = estimated_pos.copy()
                x_plus[i] += epsilon
                residuals_plus = np.array(trilateration_error(x_plus, current_anchors, current_measurements))
                J[:, i] = (residuals_plus - residuals) / epsilon
            
            # Calculate projection matrix
            try:
                P = J @ inv(J.T @ current_W @ J) @ J.T @ current_W
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                P = J @ np.linalg.pinv(J.T @ current_W @ J) @ J.T @ current_W
            
            # Calculate weighted sum of squared errors (WSSE)
            S = current_W @ (np.eye(len(current_measurements)) - P)
            WSSE = residuals.T @ S @ residuals
            
            # Calculate degrees of freedom
            dof = len(current_measurements) - n_states
            
            if dof <= 0:
                break  # Not enough measurements for outlier detection
            
            # Chi-squared test threshold
            TD = stats.chi2.ppf(1 - alpha, dof)
            
            # Check if WSSE exceeds threshold
            if WSSE <= TD:
                break  # No outliers detected, exit loop
            
            # Find measurement with largest contribution to WSSE
            # We use the absolute value of weighted residuals as a simple heuristic
            weighted_residuals = np.abs(current_W @ residuals)
            worst_measurement_idx = np.argmax(weighted_residuals)
            
            # Remove the worst measurement
            removed_global_idx = included_measurements[worst_measurement_idx]
            included_measurements.pop(worst_measurement_idx)
            
            print(f"Iteration {iteration + 1}: Removed measurement from anchor {removed_global_idx + 1}, WSSE = {WSSE:.4f}, TD = {TD:.4f}")
            
            # Check if we have enough measurements left
            if len(included_measurements) <= n_states:
                print("Warning: Too few measurements remaining for estimation")
                break
                
        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {e}")
            return np.array([np.nan, np.nan])
    
    # Final estimation with remaining measurements
    if len(included_measurements) >= n_states:
        final_anchors = anchors[included_measurements]
        final_measurements = [measurements[i] for i in included_measurements]
        try:
            result = least_squares(trilateration_error, x0, args=(final_anchors, final_measurements))
            return result.x, included_measurements
        except:
            return np.array([np.nan, np.nan]), []
    else:
        return np.array([np.nan, np.nan]), []

def estimate_position(anchors, measurements):
    """Estimate position using nonlinear least squares (original version without outlier rejection)"""
    # Initial guess (center of anchors)
    x0 = np.mean(anchors, axis=0)
    
    try:
        result = least_squares(trilateration_error, x0, args=(anchors, measurements))
        return result.x
    except:
        return np.array([np.nan, np.nan])

# Perform 1000 experiments
estimated_positions = []
estimated_positions_with_or = []  # With outlier rejection
bias_anchor_indices = []
bias_values = []
remaining_anchors_list = []  # Track which anchors were used in final estimation

for i in range(1000):
    # Randomly select one anchor to add bias
    bias_anchor_idx = random.randint(0, 3)
    bias_anchor = anchors[bias_anchor_idx]
    bias_anchor_indices.append(bias_anchor_idx)
    
    # Generate random bias value (|bias| ≤ 20cm)
    bias_value = random.uniform(-1.5, 1.5)
    bias_values.append(bias_value)
    
    # Measure distances to all anchors
    measurements = []
    for anchor in anchors:
        dist = measure_distance(true_position, anchor, bias_anchor, bias_value)
        measurements.append(dist)
    
    # Estimate position with original method
    estimated_pos = estimate_position(anchors, measurements)
    estimated_positions.append(estimated_pos)
    
    # Estimate position with outlier rejection
    estimated_pos_or, remaining_anchors = estimate_position_with_outlier_rejection(anchors, measurements)
    estimated_positions_with_or.append(estimated_pos_or)
    remaining_anchors_list.append(remaining_anchors)

# Convert to numpy arrays
estimated_positions = np.array(estimated_positions)
estimated_positions_with_or = np.array(estimated_positions_with_or)
bias_anchor_indices = np.array(bias_anchor_indices)
bias_values = np.array(bias_values)

# Filter out invalid estimates for both methods
valid_mask = ~np.isnan(estimated_positions).any(axis=1)
valid_estimates = estimated_positions[valid_mask]

valid_mask_or = ~np.isnan(estimated_positions_with_or).any(axis=1)
valid_estimates_or = estimated_positions_with_or[valid_mask_or]
valid_bias_anchors_or = bias_anchor_indices[valid_mask_or]
valid_bias_values_or = bias_values[valid_mask_or]
valid_bias_values = bias_values[valid_mask]

# Calculate positioning errors
errors = np.linalg.norm(valid_estimates - true_position, axis=1)
errors_or = np.linalg.norm(valid_estimates_or - true_position, axis=1)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: All estimated positions
ax1.scatter(anchors[:, 0], anchors[:, 1], c='red', s=100, marker='^', label='Anchors')
ax1.scatter(true_position[0], true_position[1], c='green', s=150, marker='*', label='True Position')
ax1.scatter(valid_estimates[:, 0], valid_estimates[:, 1], c='blue', alpha=0.5, s=10, label='Estimated Positions')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('UWB Positioning with Anchor Bias (1000 trials)')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Plot 2: Color by which anchor had bias
colors = ['red', 'green', 'blue', 'purple']
for i in range(4):
    mask = valid_bias_anchors_or == i
    ax2.scatter(valid_estimates[mask, 0], valid_estimates[mask, 1], 
               c=colors[i], alpha=0.6, s=10, label=f'Anchor {i+1} biased')

ax2.scatter(anchors[:, 0], anchors[:, 1], c='black', s=100, marker='^', label='Anchors')
ax2.scatter(true_position[0], true_position[1], c='orange', s=150, marker='*', label='True Position')

ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Estimated Positions Colored by Biased Anchor')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()

# Create figure with subplots for probability curves
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Probability Density Function (PDF)
ax1.hist(errors_or, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
# Add KDE curve
kde = stats.gaussian_kde(errors_or)
x_range = np.linspace(0, np.max(errors_or), 200)
ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax1.set_xlabel('Positioning Error (m)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Probability Density Function (PDF)\nof Positioning Errors')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Cumulative Distribution Function (CDF)
sorted_errors = np.sort(errors_or)
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
ax2.plot(sorted_errors, cdf, 'b-', linewidth=2)
ax2.set_xlabel('Positioning Error (m)')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('Cumulative Distribution Function (CDF)\nof Positioning Errors')
ax2.grid(True, alpha=0.3)

# Add some key percentiles to CDF plot
percentiles = [50, 68, 80, 90, 95, 99]
for p in percentiles:
    error_p = np.percentile(errors, p)
    ax2.axvline(x=error_p, color='red', linestyle='--', alpha=0.7)
    ax2.text(error_p, 0.1, f'{p}%: {error_p:.3f}m', 
             rotation=90, ha='right', va='bottom')

# Plot 3: Error vs Bias Magnitude scatter plot
ax3.scatter(np.abs(valid_bias_values), errors, alpha=0.5, s=20)
ax3.set_xlabel('Bias Magnitude (m)')
ax3.set_ylabel('Positioning Error (m)')
ax3.set_title('Positioning Error vs Bias Magnitude')
ax3.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(np.abs(valid_bias_values), errors, 1)
p = np.poly1d(z)
x_trend = np.linspace(0, np.max(np.abs(valid_bias_values)), 100)
ax3.plot(x_trend, p(x_trend), "r--", linewidth=2, 
         label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
ax3.legend()

plt.tight_layout()

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: All estimated positions (original method)
ax1.scatter(anchors[:, 0], anchors[:, 1], c='red', s=100, marker='^', label='Anchors')
ax1.scatter(true_position[0], true_position[1], c='green', s=150, marker='*', label='True Position')
ax1.scatter(valid_estimates[:, 0], valid_estimates[:, 1], c='blue', alpha=0.5, s=10, label='Estimated Positions')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Original Method: UWB Positioning with Anchor Bias (1000 trials)')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Plot 2: All estimated positions (with outlier rejection)
ax2.scatter(anchors[:, 0], anchors[:, 1], c='red', s=100, marker='^', label='Anchors')
ax2.scatter(true_position[0], true_position[1], c='green', s=150, marker='*', label='True Position')
ax2.scatter(valid_estimates_or[:, 0], valid_estimates_or[:, 1], c='blue', alpha=0.5, s=10, label='Estimated Positions')

ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('With Outlier Rejection: UWB Positioning with Anchor Bias')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

# Plot 3: Error comparison
methods = ['Original', 'With Outlier Rejection']
error_means = [np.mean(errors), np.mean(errors_or)]
error_stds = [np.std(errors), np.std(errors_or)]

bars = ax3.bar(methods, error_means, yerr=error_stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
ax3.set_ylabel('Positioning Error (m)')
ax3.set_title('Comparison of Positioning Errors')
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars, error_means, error_stds):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
             f'{mean:.3f}±{std:.3f}m', ha='center', va='bottom')

# Plot 4: Success rate comparison
success_rates = [len(valid_estimates)/1000*100, len(valid_estimates_or)/1000*100]
bars_success = ax4.bar(methods, success_rates, alpha=0.7, color=['lightgreen', 'lightyellow'])
ax4.set_ylabel('Success Rate (%)')
ax4.set_title('Comparison of Success Rates')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, rate in zip(bars_success, success_rates):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%', ha='center', va='bottom')

plt.tight_layout()

# Display comprehensive statistics
print("=" * 80)
print("POSITIONING ERROR STATISTICS COMPARISON")
print("=" * 80)
print(f"Total experiments: 1000")
print(f"Original method - Valid position estimates: {len(valid_estimates)}")
print(f"With outlier rejection - Valid position estimates: {len(valid_estimates_or)}")
print(f"Original method - Success rate: {len(valid_estimates)/1000*100:.1f}%")
print(f"With outlier rejection - Success rate: {len(valid_estimates_or)/1000*100:.1f}%")

print(f"\nError Statistics - Original Method:")
print(f"Mean error: {np.mean(errors):.4f} m")
print(f"Median error: {np.median(errors):.4f} m")
print(f"Standard deviation: {np.std(errors):.4f} m")

print(f"\nError Statistics - With Outlier Rejection:")
print(f"Mean error: {np.mean(errors_or):.4f} m")
print(f"Median error: {np.median(errors_or):.4f} m")
print(f"Standard deviation: {np.std(errors_or):.4f} m")

# Calculate improvement
improvement_mean = (np.mean(errors) - np.mean(errors_or)) / np.mean(errors) * 100
improvement_std = (np.std(errors) - np.std(errors_or)) / np.std(errors) * 100

print(f"\nImprovement with outlier rejection:")
print(f"Mean error improvement: {improvement_mean:.1f}%")
print(f"Standard deviation improvement: {improvement_std:.1f}%")

# Analyze which anchors were most frequently removed
if remaining_anchors_list:
    anchor_removal_count = np.zeros(4)
    for remaining_anchors in remaining_anchors_list:
        for i in range(4):
            if i not in remaining_anchors:
                anchor_removal_count[i] += 1
    
    print(f"\nAnchor removal statistics:")
    for i in range(4):
        print(f"Anchor {i+1} was removed in {anchor_removal_count[i]:.0f} experiments ({anchor_removal_count[i]/1000*100:.1f}%)")

plt.show()