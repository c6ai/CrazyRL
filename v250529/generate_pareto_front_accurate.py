"""Script to generate a Pareto front visualization similar to the one in the repository."""
import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("v250529", exist_ok=True)

# Define data points similar to the original image
# The original has negative values for "Close to target" and positive values for "Far from others"
close_to_target = np.array([
    -38, -35, -32, -30, -28, -26, -25, -24, -23, -20, -19, -15, -10, -5, 0
])

far_from_others = np.array([
    13000, 12500, 12300, 11500, 11000, 10500, 10400, 10300, 9700, 8900, 8800, 6700, 5900, 4500, 1800
])

# Create the figure
plt.figure(figsize=(10, 8))

# Plot the Pareto front
plt.scatter(close_to_target, far_from_others, s=100, color='#5CB5FF', alpha=0.9)

# Connect the points with a dashed line
plt.plot(close_to_target, far_from_others, 'k--', alpha=0.7)

# Add labels and grid
plt.ylabel("Far from others", fontsize=16)
plt.xlabel("Close to target", fontsize=16)
plt.title("Pareto Front of Multi-Objective Policies", fontsize=18)
plt.grid(alpha=0.25)

# Save the figure
plt.tight_layout()
plt.savefig("v250529/pareto_front_accurate.png", dpi=600)

print("Accurate Pareto front visualization generated and saved to v250529/pareto_front_accurate.png")