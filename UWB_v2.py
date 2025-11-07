import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import least_squares
from scipy import stats

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
    bias_value = random.uniform(-1.5, 1.5)
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

plt.show()