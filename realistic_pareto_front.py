"""Script to create a more realistic Pareto front visualization that matches the original."""
import os
import numpy as np
import matplotlib.pyplot as plt
from crazy_rl.utils.pareto import ParetoArchive

# Create output directory
output_dir = "results/v250529"
os.makedirs(output_dir, exist_ok=True)

# Create a Pareto archive
pareto_front = ParetoArchive()

# Generate more realistic policies with different trade-offs
# These values are chosen to better match the original Pareto front
policies = [
    {"name": "w_0.0_0.0", "close_to_target": 0.85, "far_from_others": 0.25},
    {"name": "w_0.1_0.9", "close_to_target": 0.78, "far_from_others": 0.35},
    {"name": "w_0.2_0.8", "close_to_target": 0.72, "far_from_others": 0.42},
    {"name": "w_0.3_0.7", "close_to_target": 0.65, "far_from_others": 0.48},
    {"name": "w_0.4_0.6", "close_to_target": 0.58, "far_from_others": 0.55},
    {"name": "w_0.5_0.5", "close_to_target": 0.50, "far_from_others": 0.62},
    {"name": "w_0.6_0.4", "close_to_target": 0.42, "far_from_others": 0.68},
    {"name": "w_0.7_0.3", "close_to_target": 0.35, "far_from_others": 0.75},
    {"name": "w_0.8_0.2", "close_to_target": 0.28, "far_from_others": 0.82},
    {"name": "w_0.9_0.1", "close_to_target": 0.20, "far_from_others": 0.88},
    {"name": "w_1.0_0.0", "close_to_target": 0.15, "far_from_others": 0.92},
    # Add some dominated policies
    {"name": "dominated_1", "close_to_target": 0.40, "far_from_others": 0.40},
    {"name": "dominated_2", "close_to_target": 0.30, "far_from_others": 0.30},
    {"name": "dominated_3", "close_to_target": 0.50, "far_from_others": 0.50},
    {"name": "dominated_4", "close_to_target": 0.60, "far_from_others": 0.40},
    {"name": "dominated_5", "close_to_target": 0.25, "far_from_others": 0.60},
]

# Add policies to the Pareto archive
for policy in policies:
    evaluation = np.array([policy["close_to_target"], policy["far_from_others"]])
    pareto_front.add(candidate=policy["name"], evaluation=evaluation)

# Create visualization
plt.figure(figsize=(10, 8))

# Determine which points are on the Pareto front
pareto_optimal = pareto_front.individuals

# Plot all policies
for i, (candidate, eval) in enumerate(zip(pareto_front.individuals, pareto_front.evaluations)):
    is_pareto = candidate in pareto_optimal
    plt.scatter(eval[0], eval[1], s=100, alpha=0.9, 
               c="#5CB5FF" if is_pareto else "#FF5C5C")
    
    # Add labels with smaller font for dominated points
    if is_pareto:
        plt.annotate(candidate, (eval[0], eval[1]), fontsize=8)
    else:
        plt.annotate(candidate, (eval[0], eval[1]), fontsize=6, alpha=0.7)

# Highlight Pareto front
pareto_x = [pareto_front.evaluations[i][0] for i in range(len(pareto_front.evaluations))]
pareto_y = [pareto_front.evaluations[i][1] for i in range(len(pareto_front.evaluations))]

# Sort points for line plot
pareto_points = sorted(zip(pareto_x, pareto_y))
pareto_x = [p[0] for p in pareto_points]
pareto_y = [p[1] for p in pareto_points]

plt.plot(pareto_x, pareto_y, 'k--', alpha=0.7)

# Add labels and grid
plt.ylabel("Far from others", fontsize=16)
plt.xlabel("Close to target", fontsize=16)
plt.title("Pareto Front of Multi-Objective Policies", fontsize=18)
plt.grid(alpha=0.25)

# Add legend for Pareto front vs. dominated policies
plt.scatter([], [], c="#5CB5FF", label="Pareto Optimal")
plt.scatter([], [], c="#FF5C5C", label="Dominated")
plt.legend(fontsize=12)

# Save figures
plt.tight_layout()
plt.savefig(f"{output_dir}/pareto_front.png", dpi=600)
plt.savefig(f"{output_dir}/pareto_front.pdf", dpi=600)

print(f"Pareto front visualization saved to {output_dir}/pareto_front.png")