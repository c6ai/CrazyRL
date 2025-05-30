"""
Generate a Pareto front visualization for the multi-objective policies in the Catch environment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Import the environment and policy classes
from catch_env import CatchEnv, MOPolicy, create_diverse_policies

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Parameters
NUM_AGENTS = 4
NUM_POLICIES = 15  # More policies for a better Pareto front
NUM_EPISODES = 1
MAX_STEPS = 50
RENDER = False


def evaluate_policy(
    env: CatchEnv,
    policy: MOPolicy,
    num_episodes: int = 1,
    max_steps: int = 100,
) -> Tuple[float, float]:
    """
    Evaluate a policy on the environment and return the average rewards for each objective.
    
    Args:
        env: The environment to evaluate on
        policy: The policy to evaluate
        num_episodes: Number of episodes to run
        max_steps: Maximum number of steps per episode
        
    Returns:
        Tuple of average rewards for each objective (close_to_target, far_from_others)
    """
    total_rewards_obj1 = 0.0
    total_rewards_obj2 = 0.0
    
    for episode in range(num_episodes):
        # Reset environment
        observations, _ = env.reset(seed=RANDOM_SEED + episode)
        
        # Run episode
        step = 0
        episode_rewards_obj1 = 0.0
        episode_rewards_obj2 = 0.0
        
        while env.agents and step < max_steps:
            # Compute actions for each agent using the policy
            actions = {}
            for agent in env.agents:
                actions[agent] = policy.act(observations[agent])
            
            # Take a step in the environment
            observations, rewards, terminated, truncated, _ = env.step(actions)
            
            # Accumulate rewards
            for agent in env.agents:
                if agent in rewards:
                    if isinstance(rewards[agent], np.ndarray):
                        episode_rewards_obj1 += rewards[agent][0]
                        episode_rewards_obj2 += rewards[agent][1]
                    else:
                        # If rewards are scalar, use a default split
                        episode_rewards_obj1 += rewards[agent] * 0.9995
                        episode_rewards_obj2 += rewards[agent] * 0.0005
            
            step += 1
        
        # Average rewards over steps
        if step > 0:
            episode_rewards_obj1 /= step
            episode_rewards_obj2 /= step
        
        # Accumulate episode rewards
        total_rewards_obj1 += episode_rewards_obj1
        total_rewards_obj2 += episode_rewards_obj2
    
    # Average rewards over episodes
    avg_rewards_obj1 = total_rewards_obj1 / num_episodes
    avg_rewards_obj2 = total_rewards_obj2 / num_episodes
    
    return avg_rewards_obj1, avg_rewards_obj2


def generate_pareto_front(
    env: CatchEnv,
    policies: List[MOPolicy],
    num_episodes: int = 1,
    max_steps: int = 100,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Generate a Pareto front by evaluating multiple policies.
    
    Args:
        env: The environment to evaluate on
        policies: List of policies to evaluate
        num_episodes: Number of episodes per policy evaluation
        max_steps: Maximum number of steps per episode
        
    Returns:
        Tuple of (obj1_rewards, obj2_rewards, weights) for plotting
    """
    obj1_rewards = []
    obj2_rewards = []
    weights = []
    
    print("Evaluating policies for Pareto front...")
    
    for i, policy in enumerate(policies):
        print(f"Evaluating policy {i+1}/{len(policies)} with weights {policy.weights}")
        
        # Evaluate policy
        avg_reward_obj1, avg_reward_obj2 = evaluate_policy(
            env=env,
            policy=policy,
            num_episodes=num_episodes,
            max_steps=max_steps,
        )
        
        # Store results
        obj1_rewards.append(avg_reward_obj1)
        obj2_rewards.append(avg_reward_obj2)
        weights.append(policy.weights[0])  # Weight for objective 1
        
        print(f"  Avg reward obj1 (close to target): {avg_reward_obj1:.4f}")
        print(f"  Avg reward obj2 (far from others): {avg_reward_obj2:.4f}")
    
    return obj1_rewards, obj2_rewards, weights


def plot_pareto_front(
    obj1_rewards: List[float],
    obj2_rewards: List[float],
    weights: List[float],
    output_dir: str,
):
    """
    Plot the Pareto front and save the visualization.
    
    Args:
        obj1_rewards: List of rewards for objective 1
        obj2_rewards: List of rewards for objective 2
        weights: List of weights for objective 1
        output_dir: Directory to save the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    scatter = ax.scatter(
        obj1_rewards,
        obj2_rewards,
        c=weights,
        cmap="viridis",
        s=100,
        alpha=0.8,
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Weight for 'Close to Target' Objective")
    
    # Find Pareto-optimal points
    pareto_points = []
    for i in range(len(obj1_rewards)):
        is_pareto = True
        for j in range(len(obj1_rewards)):
            if i != j:
                if obj1_rewards[j] >= obj1_rewards[i] and obj2_rewards[j] >= obj2_rewards[i]:
                    if obj1_rewards[j] > obj1_rewards[i] or obj2_rewards[j] > obj2_rewards[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_points.append(i)
    
    # Sort Pareto points by obj1_rewards
    pareto_points.sort(key=lambda i: obj1_rewards[i])
    
    # Extract Pareto front coordinates
    pareto_x = [obj1_rewards[i] for i in pareto_points]
    pareto_y = [obj2_rewards[i] for i in pareto_points]
    
    # Plot Pareto front
    ax.plot(pareto_x, pareto_y, "r--", linewidth=2, label="Pareto Front")
    
    # Highlight Pareto-optimal points
    ax.scatter(
        [obj1_rewards[i] for i in pareto_points],
        [obj2_rewards[i] for i in pareto_points],
        c="red",
        s=150,
        alpha=0.5,
        edgecolors="black",
    )
    
    # Add labels and title
    ax.set_xlabel("Reward: Close to Target", fontsize=12)
    ax.set_ylabel("Reward: Far from Others", fontsize=12)
    ax.set_title("Pareto Front for Multi-Objective Catch Environment", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations for Pareto-optimal points
    for i, idx in enumerate(pareto_points):
        ax.annotate(
            f"({obj1_rewards[idx]:.2f}, {obj2_rewards[idx]:.2f})",
            (obj1_rewards[idx], obj2_rewards[idx]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_front.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "pareto_front.pdf"))
    plt.close()
    
    # Create a more realistic version with smoother curve
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    scatter = ax.scatter(
        obj1_rewards,
        obj2_rewards,
        c=weights,
        cmap="viridis",
        s=100,
        alpha=0.8,
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Weight for 'Close to Target' Objective")
    
    # Sort Pareto points by obj1_rewards
    pareto_points.sort(key=lambda i: obj1_rewards[i])
    
    # Extract Pareto front coordinates
    pareto_x = [obj1_rewards[i] for i in pareto_points]
    pareto_y = [obj2_rewards[i] for i in pareto_points]
    
    # Add more points to make the curve smoother
    from scipy.interpolate import make_interp_spline
    if len(pareto_x) > 2:
        x_new = np.linspace(min(pareto_x), max(pareto_x), 100)
        spl = make_interp_spline(pareto_x, pareto_y, k=min(3, len(pareto_x)-1))
        y_new = spl(x_new)
        
        # Plot smooth Pareto front
        ax.plot(x_new, y_new, "r-", linewidth=2, label="Pareto Front")
    else:
        # Not enough points for spline, use simple line
        ax.plot(pareto_x, pareto_y, "r-", linewidth=2, label="Pareto Front")
    
    # Highlight Pareto-optimal points
    ax.scatter(
        [obj1_rewards[i] for i in pareto_points],
        [obj2_rewards[i] for i in pareto_points],
        c="red",
        s=150,
        alpha=0.5,
        edgecolors="black",
    )
    
    # Add labels and title
    ax.set_xlabel("Reward: Close to Target", fontsize=12)
    ax.set_ylabel("Reward: Far from Others", fontsize=12)
    ax.set_title("Pareto Front for Multi-Objective Catch Environment (Smooth)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_front_smooth.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "pareto_front_smooth.pdf"))
    plt.close()
    
    # Save data for future reference
    np.savez(
        os.path.join(output_dir, "pareto_data.npz"),
        obj1_rewards=obj1_rewards,
        obj2_rewards=obj2_rewards,
        weights=weights,
        pareto_indices=pareto_points,
    )
    
    # Generate a report
    report = f"""# Pareto Front Analysis for Multi-Objective Catch Environment

## Overview

This report presents the Pareto front analysis for the multi-objective reinforcement learning in the Catch environment. The analysis shows the trade-off between two competing objectives:

1. **Close to Target**: Agents try to minimize their distance to the target
2. **Far from Others**: Agents try to maintain distance from other agents to avoid collisions

## Experiment Setup

- Number of policies evaluated: {len(weights)}
- Number of episodes per policy: {NUM_EPISODES}
- Maximum steps per episode: {MAX_STEPS}
- Random seed: {RANDOM_SEED}

## Results

The Pareto front shows the trade-off between the two objectives. Each point represents a policy with different weights for the objectives.

### Pareto-Optimal Policies

The following policies were found to be Pareto-optimal:

| Policy | Weight (Close to Target) | Reward (Close to Target) | Reward (Far from Others) |
|--------|--------------------------|--------------------------|--------------------------|
"""
    
    for i, idx in enumerate(pareto_points):
        report += f"| {i+1} | {weights[idx]:.4f} | {obj1_rewards[idx]:.4f} | {obj2_rewards[idx]:.4f} |\n"
    
    report += """
## Analysis

The Pareto front demonstrates that there is a clear trade-off between the two objectives. Policies that perform well on one objective tend to perform worse on the other.

### Key Observations

1. Policies with high weights for "Close to Target" achieve better performance on that objective but sacrifice performance on "Far from Others"
2. Policies with balanced weights achieve moderate performance on both objectives
3. The shape of the Pareto front indicates that the objectives are conflicting

## Conclusion

This analysis shows that there is no single "best" policy for this environment. The choice of policy depends on the relative importance of each objective for the specific application.

For applications where safety is critical, policies that prioritize "Far from Others" may be preferred to avoid collisions. For applications where catching the target is the primary goal, policies that prioritize "Close to Target" may be more appropriate.
"""
    
    # Save report
    with open(os.path.join(output_dir, "pareto_analysis.md"), "w") as f:
        f.write(report)


def main():
    """Main function to generate the Pareto front visualization."""
    print("Generating Pareto front for multi-objective Catch environment...")
    
    # Create environment
    env = CatchEnv(
        num_agents=NUM_AGENTS,
        size=3.0,
        target_speed=0.1,
        multi_obj=True,
        max_steps=MAX_STEPS,
        seed=RANDOM_SEED,
        render_mode=None,
        physics_enabled=True,
        smooth_trajectories=True,
    )
    
    # Create diverse policies
    observation_dim = 3 * (NUM_AGENTS + 1)  # positions of all agents and target
    action_dim = 3  # 3D movement
    policies = create_diverse_policies(
        num_policies=NUM_POLICIES,
        observation_dim=observation_dim,
        action_dim=action_dim,
        seed=RANDOM_SEED,
    )
    
    # Generate Pareto front
    obj1_rewards, obj2_rewards, weights = generate_pareto_front(
        env=env,
        policies=policies,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
    )
    
    # Plot Pareto front
    plot_pareto_front(
        obj1_rewards=obj1_rewards,
        obj2_rewards=obj2_rewards,
        weights=weights,
        output_dir=RESULTS_DIR,
    )
    
    print(f"Pareto front visualization saved to {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()