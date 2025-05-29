"""Simple script to visualize a Pareto front for the Catch environment."""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from crazy_rl.utils.pareto import ParetoArchive


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/v250529",
                        help="Directory to save output files")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a Pareto archive
    pareto_front = ParetoArchive()
    
    # Generate some sample policies with different trade-offs
    # These are just example values - in a real scenario, these would come from trained policies
    policies = [
        {"name": "policy_0", "close_to_target": 0.85, "far_from_others": 0.25},
        {"name": "policy_1", "close_to_target": 0.75, "far_from_others": 0.45},
        {"name": "policy_2", "close_to_target": 0.65, "far_from_others": 0.60},
        {"name": "policy_3", "close_to_target": 0.55, "far_from_others": 0.70},
        {"name": "policy_4", "close_to_target": 0.45, "far_from_others": 0.75},
        {"name": "policy_5", "close_to_target": 0.35, "far_from_others": 0.80},
        {"name": "policy_6", "close_to_target": 0.25, "far_from_others": 0.85},
        # Add some dominated policies
        {"name": "policy_7", "close_to_target": 0.40, "far_from_others": 0.40},
        {"name": "policy_8", "close_to_target": 0.30, "far_from_others": 0.30},
        {"name": "policy_9", "close_to_target": 0.50, "far_from_others": 0.50},
    ]
    
    # Add policies to the Pareto archive
    for policy in policies:
        evaluation = np.array([policy["close_to_target"], policy["far_from_others"]])
        pareto_front.add(candidate=policy["name"], evaluation=evaluation)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Determine which points are on the Pareto front
    # Since ParetoArchive already filters dominated points, all points in the archive are on the Pareto front
    pareto_optimal = pareto_front.individuals
    
    # Plot all policies
    for i, (candidate, eval) in enumerate(zip(pareto_front.individuals, pareto_front.evaluations)):
        plt.scatter(eval[0], eval[1], s=100, label=candidate, alpha=0.9, 
                   c="#5CB5FF" if candidate in pareto_optimal else "#FF5C5C")
        plt.annotate(candidate, (eval[0], eval[1]), fontsize=8)
    
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
    plt.savefig(f"{args.output_dir}/pareto_front.png", dpi=600)
    plt.savefig(f"{args.output_dir}/pareto_front.pdf", dpi=600)
    
    # Print results
    print("\nPareto Front Policies:")
    for ind, eval in zip(pareto_front.individuals, pareto_front.evaluations):
        print(f"  {ind}: Close to target = {eval[0]:.4f}, Far from others = {eval[1]:.4f}")
    
    print(f"\nPareto front visualization saved to {args.output_dir}/pareto_front.png")


if __name__ == "__main__":
    main()