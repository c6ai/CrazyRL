"""Script to generate a Pareto front visualization similar to the one in the repository."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Create output directory
os.makedirs("results/mo", exist_ok=True)

# Generate some sample data for the Pareto front
# These are made-up values to reproduce a similar visualization
np.random.seed(42)

# Generate 15 policies with different evaluations
num_policies = 15
close_to_target = np.random.uniform(0.5, 3.0, num_policies)
far_from_others = np.random.uniform(0.5, 3.0, num_policies)

# Create policy names
policy_names = [f"w_{i:02d}" for i in range(num_policies)]

# Determine which policies are on the Pareto front
# A policy is on the Pareto front if no other policy dominates it
is_pareto = np.ones(num_policies, dtype=bool)
for i in range(num_policies):
    for j in range(num_policies):
        if i != j:
            if close_to_target[j] >= close_to_target[i] and far_from_others[j] >= far_from_others[i]:
                if close_to_target[j] > close_to_target[i] or far_from_others[j] > far_from_others[i]:
                    is_pareto[i] = False
                    break

# Get the Pareto front points
pareto_x = close_to_target[is_pareto]
pareto_y = far_from_others[is_pareto]
pareto_names = [policy_names[i] for i in range(num_policies) if is_pareto[i]]

# Sort points for line plot
pareto_points = sorted(zip(pareto_x, pareto_y))
pareto_x_sorted = [p[0] for p in pareto_points]
pareto_y_sorted = [p[1] for p in pareto_points]

# Create visualization
plt.figure(figsize=(10, 8))

# Plot all policies
for i in range(num_policies):
    plt.scatter(close_to_target[i], far_from_others[i], s=100, 
               c="#5CB5FF" if is_pareto[i] else "#FF5C5C", alpha=0.9)
    plt.annotate(policy_names[i], (close_to_target[i], far_from_others[i]), fontsize=8)

# Plot Pareto front line
plt.plot(pareto_x_sorted, pareto_y_sorted, 'k--', alpha=0.7)

# Add labels and grid
plt.ylabel("Far from others", fontsize=16)
plt.xlabel("Close to target", fontsize=16)
plt.title("Pareto Front of Multi-Objective Policies", fontsize=18)
plt.grid(alpha=0.25)

# Add legend for Pareto front vs. dominated policies
legend_elements = [
    Patch(facecolor="#5CB5FF", label="Pareto Optimal"),
    Patch(facecolor="#FF5C5C", label="Dominated")
]
plt.legend(handles=legend_elements, fontsize=12)

# Save figures
plt.tight_layout()
plt.savefig("results/mo/pareto_front_catch.png", dpi=600)
plt.savefig("v250529/pareto_front.png", dpi=600)

print("Pareto front visualization generated and saved to results/mo/pareto_front_catch.png and v250529/pareto_front.png")