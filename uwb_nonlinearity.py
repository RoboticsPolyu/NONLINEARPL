import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import least_squares
from scipy import stats

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'  # For mathematical symbols

# Optional: Set other font-related parameters for better appearance
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Set random seed for reproducibility
np.random.seed(42)

# Define four anchor positions (x, y)
anchors = np.array([
    [0, 0],    # Anchor 1
    [10, 0],   # Anchor 2
    [10, 5],  # Anchor 3
    [0, 5]    # Anchor 4
])

noise_sigma = 0.2
noise_mean = 0.0
bias_magnitude = 1.5
num_anchors = 4

# Define the true tag position
true_position = np.array([5.0, 3.0])

# Weight matrix (inverse of covariance)
W = np.eye(num_anchors) / (noise_sigma ** 2)

# Distance measurement function with noise and bias
def measure_distance(true_pos, anchor_pos, bias_anchor=None, bias_value=0):
    """Calculate distance measurement with noise and optional bias"""
    true_dist = np.linalg.norm(true_pos - anchor_pos)
    # Add Gaussian noise (standard deviation 5cm)
    noise = np.random.normal(noise_mean, noise_sigma)
    
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

def estimate_position(anchors, measurements):
    """Estimate position using nonlinear least squares"""
    # Initial guess (center of anchors)
    x0 = np.mean(anchors, axis=0)
    
    try:
        result = least_squares(trilateration_error, x0, args=(anchors, measurements))
        return result.x
    except:
        return np.array([np.nan, np.nan])

# Perform 1000 experiments
estimated_positions = []
bias_anchor_indices = []
bias_values = []

for i in range(1000):
    # Randomly select one anchor to add bias
    bias_anchor_idx = random.randint(0, 3)
    bias_anchor = anchors[bias_anchor_idx]
    bias_anchor_indices.append(bias_anchor_idx)
    
    # Generate random bias value (|bias| ≤ 20cm)
    bias_value = random.uniform(-bias_magnitude, bias_magnitude)
    bias_values.append(bias_value)
    
    # Measure distances to all anchors
    measurements = []
    for anchor in anchors:
        dist = measure_distance(true_position, anchor, bias_anchor, bias_value)
        measurements.append(dist)
    
    # Estimate position
    estimated_pos = estimate_position(anchors, measurements)
    estimated_positions.append(estimated_pos)

# Convert to numpy arrays
estimated_positions = np.array(estimated_positions)
bias_anchor_indices = np.array(bias_anchor_indices)
bias_values = np.array(bias_values)

# Filter out invalid estimates
valid_mask = ~np.isnan(estimated_positions).any(axis=1)
valid_estimates = estimated_positions[valid_mask]
valid_bias_anchors = bias_anchor_indices[valid_mask]
valid_bias_values = bias_values[valid_mask]

# Calculate positioning errors
errors = np.linalg.norm(valid_estimates - true_position, axis=1)

# Create figure with subplots
fig, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

# Plot 1: All estimated positions
# ax1.scatter(anchors[:, 0], anchors[:, 1], c='red', s=100, marker='^', label='Anchors')
# ax1.scatter(true_position[0], true_position[1], c='green', s=150, marker='*', label='True Position')
# ax1.scatter(valid_estimates[:, 0], valid_estimates[:, 1], c='blue', alpha=0.5, s=10, label='Estimated Positions')

# ax1.set_xlabel('X (m)')
# ax1.set_ylabel('Y (m)')
# ax1.set_title('UWB Positioning with Anchor Bias (1000 trials)')
# ax1.legend()
# ax1.grid(True)
# ax1.axis('equal')

# Plot 2: Color by which anchor had bias
colors = ['red', 'green', 'blue', 'purple']
for i in range(4):
    mask = valid_bias_anchors == i
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
plt.savefig('uwb_biased_anchor_sims.pdf', dpi=300, bbox_inches='tight')

plt.tight_layout()

# Create figure with subplots for probability curves
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Probability Density Function (PDF)
ax1.hist(errors, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
# Add KDE curve
kde = stats.gaussian_kde(errors)
x_range = np.linspace(0, np.max(errors), 200)
ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax1.set_xlabel('Positioning Error (m)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Probability Density Function (PDF)\nof Positioning Errors')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Cumulative Distribution Function (CDF)
sorted_errors = np.sort(errors)
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

# Create additional figure for detailed statistics
fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 4: Error distribution by biased anchor
colors = ['red', 'green', 'blue', 'purple']
anchor_labels = ['Anchor 1', 'Anchor 2', 'Anchor 3', 'Anchor 4']
error_by_anchor = []

for i in range(4):
    mask = valid_bias_anchors == i
    anchor_errors = errors[mask]
    error_by_anchor.append(anchor_errors)
    
    # Plot individual CDFs
    sorted_anchor_errors = np.sort(anchor_errors)
    anchor_cdf = np.arange(1, len(sorted_anchor_errors) + 1) / len(sorted_anchor_errors)
    ax4.plot(sorted_anchor_errors, anchor_cdf, color=colors[i], 
             linewidth=2, label=anchor_labels[i])

ax4.set_xlabel('Positioning Error (m)')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('CDF by Biased Anchor')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Box plot of errors by anchor
box_plot_data = [errors[valid_bias_anchors == i] for i in range(4)]
ax5.boxplot(box_plot_data, labels=anchor_labels)
ax5.set_ylabel('Positioning Error (m)')
ax5.set_title('Error Distribution by Biased Anchor')
ax5.grid(True, alpha=0.3)

plt.tight_layout()

# Display comprehensive statistics
print("=" * 60)
print("POSITIONING ERROR STATISTICS")
print("=" * 60)
print(f"Total experiments: 1000")
print(f"Valid position estimates: {len(valid_estimates)}")
print(f"Success rate: {len(valid_estimates)/1000*100:.1f}%")
print(f"\nError Statistics:")
print(f"Mean error: {np.mean(errors):.4f} m")
print(f"Median error: {np.median(errors):.4f} m")
print(f"Standard deviation: {np.std(errors):.4f} m")
print(f"Minimum error: {np.min(errors):.4f} m")
print(f"Maximum error: {np.max(errors):.4f} m")

print(f"\nKey Percentiles:")
percentiles = [50, 68, 80, 90, 95, 99, 99.9]
for p in percentiles:
    error_val = np.percentile(errors, p)
    print(f"  {p:5.1f}%: {error_val:.4f} m")

print(f"\nBias Statistics:")
print(f"Mean bias magnitude: {np.mean(np.abs(valid_bias_values)):.4f} m")
print(f"Bias standard deviation: {np.std(valid_bias_values):.4f} m")

print(f"\nCorrelation between bias magnitude and error:")
correlation = np.corrcoef(np.abs(valid_bias_values), errors)[0, 1]
print(f"Correlation coefficient: {correlation:.4f}")

# Display statistics by anchor
print(f"\nError Statistics by Biased Anchor:")
print("Anchor | Count | Mean Error | Std Error | Max Error")
print("-" * 55)
for i in range(4):
    mask = valid_bias_anchors == i
    anchor_errors = errors[mask]
    if len(anchor_errors) > 0:
        print(f"{i+1:6d} | {len(anchor_errors):5d} | {np.mean(anchor_errors):10.4f} | "
              f"{np.std(anchor_errors):9.4f} | {np.max(anchor_errors):9.4f}")

# =============================================================================
# NEW: Cost Function Landscape Visualization for Nonlinearity Analysis
# =============================================================================

# =============================================================================
# NEW: Gradient Field Visualization (without anchors)
# =============================================================================

def plot_gradient_field():
    """Plot gradient field around estimated position without anchor markers"""
    # Select a representative sample
    sample_idx = np.random.choice(len(valid_estimates), 1)[0]
    sample_est = valid_estimates[sample_idx]
    
    # Recreate measurements for this sample
    bias_anchor = anchors[valid_bias_anchors[sample_idx]]
    bias_value = valid_bias_values[sample_idx]
    measurements = []
    for anchor in anchors:
        dist = measure_distance(true_position, anchor, bias_anchor, bias_value)
        measurements.append(dist)
    
    # Create grid around estimated position
    x_range = np.linspace(sample_est[0] - 2.0, sample_est[0] + 2.0, 20)
    y_range = np.linspace(sample_est[1] - 2.0, sample_est[1] + 2.0, 20)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate gradient and cost
    U = np.zeros_like(X)  # Gradient x-component
    V = np.zeros_like(Y)  # Gradient y-component
    Z = np.zeros_like(X)  # Cost function values
    
    eps = 0.01  # Step size for numerical differentiation
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val, y_val = X[i, j], Y[i, j]
            Z[i, j] = cost_function(x_val, y_val, anchors, measurements)
            
            # Numerical gradient calculation
            grad_x = (cost_function(x_val + eps, y_val, anchors, measurements) - 
                     cost_function(x_val - eps, y_val, anchors, measurements)) / (2 * eps)
            grad_y = (cost_function(x_val, y_val + eps, anchors, measurements) - 
                     cost_function(x_val, y_val - eps, anchors, measurements)) / (2 * eps)
            
            U[i, j] = -grad_x  # Negative gradient direction (descent direction)
            V[i, j] = -grad_y
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Gradient field with streamlines
    magnitude = np.sqrt(U**2 + V**2)  # Gradient magnitude
    strm = ax1.streamplot(X, Y, U, V, color=magnitude, cmap='viridis', 
                         linewidth=1.5, density=2, arrowsize=1.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Gradient Field Visualization\n(Streamlines show negative gradient direction)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    fig.colorbar(strm.lines, ax=ax1, label='Gradient Magnitude')
    
    # Subplot 2: Gradient field with quiver plot
    # Normalize arrows for better visualization
    max_magnitude = np.max(magnitude)
    if max_magnitude > 0:
        U_norm = U / max_magnitude
        V_norm = V / max_magnitude
    else:
        U_norm = U
        V_norm = V
        
    quiver = ax2.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis', 
                       scale=20, width=0.005, angles='xy')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Gradient Field with Quiver Plot\n(Normalized arrow directions)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    fig.colorbar(quiver, ax=ax2, label='Gradient Magnitude')
    
    # Mark only the estimated position (no anchors)
    for ax in [ax1, ax2]:
        ax.scatter(sample_est[0], sample_est[1], c='red', s=150, marker='*', 
                  label='Estimated Position', zorder=5)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print gradient information
    print(f"\nGradient Analysis for Sample:")
    print(f"Estimated position: ({sample_est[0]:.3f}, {sample_est[1]:.3f})")
    print(f"Positioning error: {errors[sample_idx]:.4f} m")
    print(f"Max gradient magnitude: {np.max(magnitude):.4f}")
    print(f"Mean gradient magnitude: {np.mean(magnitude):.4f}")
    print(f"Gradient at estimated point: ({U[X.shape[0]//2, X.shape[1]//2]:.4f}, "
          f"{V[X.shape[0]//2, Y.shape[1]//2]:.4f})")

# =============================================================================
# NEW: Gradient Field Visualization with Contours (without anchors)
# =============================================================================

def plot_gradient_field_with_contours():
    """Plot gradient field with contour lines around estimated position without anchor markers"""
    # Select a representative sample
    sample_idx = np.random.choice(len(valid_estimates), 1)[0]
    sample_est = valid_estimates[sample_idx]
    
    # Recreate measurements for this sample
    bias_anchor = anchors[valid_bias_anchors[sample_idx]]
    bias_value = valid_bias_values[sample_idx]
    measurements = []
    for anchor in anchors:
        dist = measure_distance(true_position, anchor, bias_anchor, bias_value)
        measurements.append(dist)
    
    # Create grid around estimated position (denser grid for better contours)
    x_range = np.linspace(sample_est[0] - 2.0, sample_est[0] + 2.0, 30)
    y_range = np.linspace(sample_est[1] - 2.0, sample_est[1] + 2.0, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate gradient and cost
    U = np.zeros_like(X)  # Gradient x-component
    V = np.zeros_like(Y)  # Gradient y-component
    Z = np.zeros_like(X)  # Cost function values
    
    eps = 0.01  # Step size for numerical differentiation
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val, y_val = X[i, j], Y[i, j]
            Z[i, j] = cost_function(x_val, y_val, anchors, measurements)
            
            # Numerical gradient calculation
            grad_x = (cost_function(x_val + eps, y_val, anchors, measurements) - 
                     cost_function(x_val - eps, y_val, anchors, measurements)) / (2 * eps)
            grad_y = (cost_function(x_val, y_val + eps, anchors, measurements) - 
                     cost_function(x_val, y_val - eps, anchors, measurements)) / (2 * eps)
            
            U[i, j] = -grad_x  # Negative gradient direction (descent direction)
            V[i, j] = -grad_y
    
    # Create figure with three subplots
    fig, (ax2) = plt.subplots(1, 1, figsize= (5, 4))
    
    # Subplot 1: Gradient field with streamlines and contours
    magnitude = np.sqrt(U**2 + V**2)  # Gradient magnitude
    # strm = ax1.streamplot(X, Y, U, V, color=magnitude, cmap='viridis', 
    #                      linewidth=1.5, density=2, arrowsize=1.5)
    # # Add contour lines
    # contour1 = ax1.contour(X, Y, Z, levels=15, colors='black', linewidths=0.8, alpha=0.7)
    # ax1.clabel(contour1, inline=True, fontsize=8, fmt='%.1f')
    # ax1.set_xlabel('X (m)')
    # ax1.set_ylabel('Y (m)')
    # ax1.set_title('Gradient Field with Contours\n(Streamlines + Contour lines)')
    # ax1.grid(True, alpha=0.3)
    # ax1.axis('equal')
    # fig.colorbar(strm.lines, ax=ax1, label='Gradient Magnitude')
    
    # Subplot 2: Gradient field with quiver and contours
    # Normalize arrows for better visualization
    max_magnitude = np.max(magnitude)
    if max_magnitude > 0:
        U_norm = U / max_magnitude
        V_norm = V / max_magnitude
    else:
        U_norm = U
        V_norm = V
        
    quiver = ax2.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis', 
                       scale=20, width=0.005, angles='xy')
    # Add contour lines
    contour2 = ax2.contour(X, Y, Z, levels=15, colors='black', linewidths=0.8, alpha=0.7)
    ax2.clabel(contour2, inline=True, fontsize=8, fmt='%.1f')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Gradient Field with Contours\n(Quiver + Contour lines)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    fig.colorbar(quiver, ax=ax2, label='Gradient Magnitude')
    
    # Subplot 3: Contour plot only with gradient directions
    # contour3 = ax3.contourf(X, Y, Z, levels=20, alpha=0.8, cmap='plasma')
    # contour_lines = ax3.contour(X, Y, Z, levels=15, colors='white', linewidths=0.8, alpha=0.7)
    # ax3.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # # Add gradient arrows on contour plot (sparse for clarity)
    # skip = 3  # Show every 3rd arrow for clarity
    # ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
    #            U_norm[::skip, ::skip], V_norm[::skip, ::skip],
    #            color='white', scale=15, width=0.008)
    
    # ax3.set_xlabel('X (m)')
    # ax3.set_ylabel('Y (m)')
    # ax3.set_title('Cost Function Contours with Gradient Directions')
    # ax3.grid(True, alpha=0.3)
    # ax3.axis('equal')
    # fig.colorbar(contour3, ax=ax3, label='Cost Function Value')
    
    # Mark only the estimated position (no anchors) on all subplots
    for ax in [ax2]:
        ax.scatter(sample_est[0], sample_est[1], c='red', s=200, marker='*', 
                  edgecolors='white', linewidth=2, label='Estimated Position', zorder=5)
        ax.legend()

    plt.savefig('gradient_field.pdf', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    # Print gradient and cost function information
    print(f"\nGradient and Contour Analysis for Sample:")
    print(f"Estimated position: ({sample_est[0]:.3f}, {sample_est[1]:.3f})")
    print(f"Positioning error: {errors[sample_idx]:.4f} m")
    print(f"Cost at estimated position: {Z[X.shape[0]//2, X.shape[1]//2]:.4f}")
    print(f"Max gradient magnitude: {np.max(magnitude):.4f}")
    print(f"Mean gradient magnitude: {np.mean(magnitude):.4f}")
    print(f"Gradient at estimated point: ({U[X.shape[0]//2, X.shape[1]//2]:.6f}, "
          f"{V[X.shape[0]//2, Y.shape[1]//2]:.6f})")


def cost_function(x, y, anchors, measurements):
    """Calculate total cost at point (x,y)"""
    point = np.array([x, y])
    errors = trilateration_error(point, anchors, measurements)
    return np.sum(np.array(errors)**2)

# Select a few representative examples
sample_indices = np.random.choice(len(valid_estimates), min(9, len(valid_estimates)), replace=False)

fig3, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, sample_idx in enumerate(sample_indices):
    if idx >= len(axes):
        break
        
    sample_est = valid_estimates[sample_idx]
    sample_bias_anchor = valid_bias_anchors[sample_idx]
    
    # Get the measurements for this sample
    bias_anchor = anchors[sample_bias_anchor]
    bias_value = valid_bias_values[sample_idx]
    
    # Recreate measurements for this sample
    measurements = []
    for anchor in anchors:
        dist = measure_distance(true_position, anchor, bias_anchor, bias_value)
        measurements.append(dist)
    
    # Create grid around the estimated point
    x_range = np.linspace(sample_est[0] - 1.0, sample_est[0] + 1.0, 50)
    y_range = np.linspace(sample_est[1] - 1.0, sample_est[1] + 1.0, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate cost at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = cost_function(X[i, j], Y[i, j], anchors, measurements)
    
    # Plot contour
    contour = axes[idx].contourf(X, Y, Z, levels=20, alpha=0.8, cmap='viridis')
    axes[idx].contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.7)
    
    # Mark important points
    axes[idx].scatter(anchors[:, 0], anchors[:, 1], c='red', s=80, marker='^', label='Anchors', zorder=5)
    axes[idx].scatter(true_position[0], true_position[1], c='green', s=100, marker='*', label='True Pos', zorder=5)
    axes[idx].scatter(sample_est[0], sample_est[1], c='blue', s=80, marker='o', label='Estimated', zorder=5)
    
    axes[idx].set_xlabel('X (m)')
    axes[idx].set_ylabel('Y (m)')
    axes[idx].set_title(f'Sample {idx+1}: Anchor {sample_bias_anchor+1} biased\nError: {errors[sample_idx]:.3f}m')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axis('equal')

# Add a colorbar for the contour plot
fig3.colorbar(contour, ax=axes, shrink=0.8, label='Cost Function Value')
fig3.suptitle('Cost Function Landscape Showing Nonlinearity Around Estimated Positions', fontsize=16, y=0.98)
plt.tight_layout()

# =============================================================================
# Additional analysis: Show cost function at true position vs estimated position
# =============================================================================

print("\n" + "=" * 60)
print("COST FUNCTION ANALYSIS AT KEY POINTS")
print("=" * 60)

for idx, sample_idx in enumerate(sample_indices[:3]):  # Show first 3 samples
    sample_est = valid_estimates[sample_idx]
    sample_bias_anchor = valid_bias_anchors[sample_idx]
    
    # Recreate measurements
    bias_anchor = anchors[sample_bias_anchor]
    bias_value = valid_bias_values[sample_idx]
    measurements = []
    for anchor in anchors:
        dist = measure_distance(true_position, anchor, bias_anchor, bias_value)
        measurements.append(dist)
    
    # Calculate cost at different points
    cost_at_estimated = cost_function(sample_est[0], sample_est[1], anchors, measurements)
    cost_at_true = cost_function(true_position[0], true_position[1], anchors, measurements)
    cost_at_anchor_mean = cost_function(np.mean(anchors[:, 0]), np.mean(anchors[:, 1]), anchors, measurements)
    
    print(f"Sample {idx+1} (Anchor {sample_bias_anchor+1} biased):")
    print(f"  Cost at estimated position: {cost_at_estimated:.6f}")
    print(f"  Cost at true position:      {cost_at_true:.6f}")
    print(f"  Cost at anchors center:     {cost_at_anchor_mean:.6f}")
    print(f"  Positioning error:          {errors[sample_idx]:.4f} m")
    print()

# Call the gradient plotting function with contours
plot_gradient_field_with_contours()

plt.show()