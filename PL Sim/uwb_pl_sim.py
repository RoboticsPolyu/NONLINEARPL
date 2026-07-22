import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Number of anchors
numAnchors = 8

# Anchor positions (x, y, z)
anchorPos = np.array([
    [0, 0, 0],
    [10, 0, 0],
    [10, 10, 0],
    [0, 10, 0],
    [5, 0, 5],
    [5, 10, 5],
    [0, 5, 5],
    [10, 5, 5]
])

# Simulated moving anchor position (x, y, z)
t = np.linspace(0, 2 * np.pi, 100)
movingAnchorPos = np.column_stack([
    5 + np.cos(t),
    5 + np.sin(t),
    5 + np.sin(2 * t)
])

# Standard deviations for Gaussian noise and bias
gaussianStd = 0.05  # Standard deviation for Gaussian noise
bias = 0.3  # Bias noise

# assuming two anchor's bias <= 0.3
# Simulate distance measurements with noise
distances = np.zeros((100, numAnchors))
distances2 = np.zeros((100, numAnchors))

for i in range(100):
    if i > 50:
        for j in range(numAnchors):
            # True distance
            trueDistance = np.linalg.norm(movingAnchorPos[i, :] - anchorPos[j, :])
            
            # Add Gaussian noise and bias noise
            if j == 3:  # Note: Python uses 0-based indexing, so j=3 corresponds to the 4th anchor
                bias_val = 0.25
                noise = gaussianStd * np.random.randn() + bias_val
            elif j == 0:  # 1st anchor
                bias_val = 0.15
                noise = gaussianStd * np.random.randn() + bias_val
            else:
                noise = gaussianStd * np.random.randn()
            distances[i, j] = trueDistance + noise
    else:
        bias_val = -0.2
        for j in range(numAnchors):
            # True distance
            trueDistance = np.linalg.norm(movingAnchorPos[i, :] - anchorPos[j, :])
            
            # Add Gaussian noise and bias noise
            if j == 5 or j == 1:  # 6th and 2nd anchors
                noise = gaussianStd * np.random.randn() + bias_val
            else:
                noise = gaussianStd * np.random.randn()
            distances[i, j] = trueDistance + noise

# Estimate position using Nonlinear Optimization
estimatedPos_NLO = np.zeros((100, 3))

def objective_func(pos, dist_measurements):
    return np.sum((np.sqrt(np.sum((anchorPos - pos)**2, axis=1)) - dist_measurements)**2)

for i in range(100):
    initial_guess = np.array([0, 0, 0])
    result = minimize(objective_func, initial_guess, args=(distances[i, :],), method='BFGS')
    estimatedPos_NLO[i, :] = result.x

# Compute estimation errors
estimationErrors_NLO = np.sqrt(np.sum((movingAnchorPos - estimatedPos_NLO)**2, axis=1))

# Plot results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(movingAnchorPos[:, 0], movingAnchorPos[:, 1], movingAnchorPos[:, 2], 'r', linewidth=2, label='True Position')
ax.plot(estimatedPos_NLO[:, 0], estimatedPos_NLO[:, 1], estimatedPos_NLO[:, 2], 'g-.', linewidth=2, label='Estimated Position (NLO)')
ax.scatter(anchorPos[:, 0], anchorPos[:, 1], anchorPos[:, 2], c='k', s=50, label='Anchor Positions')
ax.legend()
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('True and Estimated Positions of Moving Anchor with Noise and Anchor Coordinates')
ax.grid(True)

# Calculate estimation errors
errors_nl = estimatedPos_NLO - movingAnchorPos

# Plotting 2D view
plt.figure(figsize=(10, 8))
plt.plot(movingAnchorPos[:, 0], movingAnchorPos[:, 1], 'r', linewidth=2, label='True Position')
plt.plot(estimatedPos_NLO[:, 0], estimatedPos_NLO[:, 1], 'g-.', linewidth=2, label='Estimated Position (NLO)')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('True and Estimated Positions of Moving Anchor with Noise and Anchor Coordinates')
plt.grid(True)

# Plot errors
plt.figure(figsize=(10, 8))
plt.plot(np.abs(errors_nl[:, 0]), 'r--', label='NL X Error', linewidth=2)
plt.plot(np.abs(errors_nl[:, 1]), 'g--', label='NL Y Error', linewidth=2)
plt.plot(np.abs(errors_nl[:, 2]), 'b--', label='NL Z Error', linewidth=2)
plt.xlabel('Sample Number')
plt.ylabel('Error (m)')
plt.title('Position Estimation Error using Nonlinear Optimization')
plt.legend()
plt.grid(True)
plt.ylim([0, 0.6])

# PLO estimation
estimatedPos_PL = np.zeros((100, 3))
optimizationValues = np.zeros(100)
point_num = 10000
PLO = np.zeros((point_num * 100, 3))

for i in range(100):
    optimizationValues[i] = 1000
    estimatedPos_PL[i, :] = 0
    for genj in range(point_num):
        # Number of dimensions
        n = 8
        
        # Initialize the vector with zeros
        b = np.zeros(n)
        
        # Randomly select two indices
        indices = np.random.choice(n, 2, replace=False)
        
        alpha = 0.10
        # Generate two random elements such that their squared sum is <= 0.09
        magnitude = np.sqrt(alpha)
        elements = np.random.randn(2)
        
        # Scale the elements to ensure the constraint is met
        if np.sum(elements**2) > alpha:
            elements = elements / np.sqrt(np.sum(elements**2)) * magnitude
        
        # Assign the non-zero elements to the randomly selected indices
        b[indices] = elements / np.random.randint(1, 101)
        
        distances2[i, :] = distances[i, :] - b
        
        def objective_func_pl(PL):
            return np.sum((np.sqrt(np.sum((anchorPos - estimatedPos_NLO[i, :] - PL)**2, axis=1)) - distances2[i, :])**2)
        
        initial_guess = np.array([0, 0, 0])
        result = minimize(objective_func_pl, initial_guess, method='BFGS')
        PLL = result.x
        value = result.fun
        
        idx = i * point_num + genj
        PLO[idx, :] = PLL
        
        # Update estimatedPos_PL with maximum absolute values
        if abs(PLL[0]) > estimatedPos_PL[i, 0]:
            estimatedPos_PL[i, 0] = abs(PLL[0])
            optimizationValues[i] = value
        
        if abs(PLL[1]) > estimatedPos_PL[i, 1]:
            estimatedPos_PL[i, 1] = abs(PLL[1])
            optimizationValues[i] = value
        
        if abs(PLL[2]) > estimatedPos_PL[i, 2]:
            estimatedPos_PL[i, 2] = abs(PLL[2])
            optimizationValues[i] = value

# Plot PL results
plt.figure(figsize=(10, 8))
plt.plot(np.abs(estimatedPos_PL[:, 0]) + 0.1, 'r-', label='PL X', linewidth=2)
plt.plot(np.abs(estimatedPos_PL[:, 1]) + 0.1, 'g-', label='PL Y', linewidth=2)
plt.plot(np.abs(estimatedPos_PL[:, 2]) + 0.15, 'b-', label='PL Z', linewidth=2)
plt.xlabel('Sample Number')
plt.ylabel('Length (m)')
plt.title('PL')
plt.legend()
plt.grid(True)

# Plot PLO scatter plots
plt.figure(figsize=(10, 8))
plt.plot(PLO[:, 0], PLO[:, 1], 'g.', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PLO XY Scatter')

plt.figure(figsize=(10, 8))
plt.plot(PLO[:, 0], PLO[:, 2], 'g.', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Z')
plt.title('PLO XZ Scatter')

plt.show()