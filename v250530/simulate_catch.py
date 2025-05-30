"""
Simulation script for the Catch environment with multi-objective reinforcement learning.
This script runs a simulation with multiple agents using different policies and
generates visualization assets.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Any, Optional

# Import the self-contained environment
from catch_env import CatchEnv, MOPolicy, create_diverse_policies

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Simulation parameters
NUM_AGENTS = 4
NUM_POLICIES = 5
NUM_EPISODES = 3
MAX_STEPS = 100
RENDER = False


def run_simulation(
    env: CatchEnv,
    policies: List[MOPolicy],
    num_episodes: int = 1,
    max_steps: int = 100,
    render: bool = False
) -> List[Dict[str, Any]]:
    """
    Run a simulation with the given environment and policies.
    
    Args:
        env: The environment to run the simulation in
        policies: List of policies to use (one per agent)
        num_episodes: Number of episodes to run
        max_steps: Maximum number of steps per episode
        render: Whether to render the environment
        
    Returns:
        List of episode data dictionaries
    """
    all_episode_data = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        observations, _ = env.reset(seed=RANDOM_SEED + episode)
        
        # Initialize episode data
        episode_data = {
            "episode": episode,
            "steps": [],
            "rewards": {agent: [] for agent in env.agent_names},
            "positions": {agent: [] for agent in env.agent_names},
            "target_positions": [],
            "distances_to_target": {agent: [] for agent in env.agent_names},
            "distances_between_agents": [],
            "terminated": {agent: False for agent in env.agent_names},
            "truncated": {agent: False for agent in env.agent_names},
        }
        
        # Store initial positions
        for agent in env.agent_names:
            episode_data["positions"][agent].append(env.agent_positions[agent].tolist())
        episode_data["target_positions"].append(env.target_position.tolist())
        
        # Run episode
        step = 0
        while env.agents and step < max_steps:
            # Compute actions for each agent using its policy
            actions = {}
            for i, agent in enumerate(env.agents):
                # Use policy index modulo number of policies to handle case where num_agents > num_policies
                policy_idx = i % len(policies)
                actions[agent] = policies[policy_idx].act(observations[agent])
            
            # Take a step in the environment
            observations, rewards, terminated, truncated, _ = env.step(actions)
            
            # Store step data
            step_data = {
                "step": step,
                "actions": {agent: actions.get(agent, np.zeros(3)).tolist() if hasattr(actions.get(agent, np.zeros(3)), 'tolist') else list(actions.get(agent, [0, 0, 0])) for agent in env.agent_names},
                "rewards": {agent: rewards.get(agent, np.zeros(2)).tolist() if isinstance(rewards.get(agent, 0), np.ndarray) else rewards.get(agent, 0) for agent in env.agent_names},
                "terminated": terminated,
                "truncated": truncated,
            }
            episode_data["steps"].append(step_data)
            
            # Store rewards
            for agent in env.agent_names:
                if agent in rewards:
                    episode_data["rewards"][agent].append(rewards[agent].tolist() if isinstance(rewards[agent], np.ndarray) else rewards[agent])
                else:
                    episode_data["rewards"][agent].append([0, 0] if env.multi_obj else 0)
            
            # Store positions
            for agent in env.agent_names:
                if agent in env.agent_positions:
                    episode_data["positions"][agent].append(env.agent_positions[agent].tolist())
                else:
                    # Use last known position if agent is terminated
                    episode_data["positions"][agent].append(episode_data["positions"][agent][-1])
            
            # Store target position
            episode_data["target_positions"].append(env.target_position.tolist())
            
            # Calculate distances to target
            for agent in env.agent_names:
                if agent in env.agent_positions:
                    dist = np.linalg.norm(env.agent_positions[agent] - env.target_position)
                    episode_data["distances_to_target"][agent].append(float(dist))
                else:
                    # Use last known distance if agent is terminated
                    episode_data["distances_to_target"][agent].append(episode_data["distances_to_target"][agent][-1] if episode_data["distances_to_target"][agent] else 0)
            
            # Calculate distances between agents
            distances = []
            for i, agent1 in enumerate(env.agent_names):
                for j, agent2 in enumerate(env.agent_names):
                    if i < j:  # Only calculate each pair once
                        if agent1 in env.agent_positions and agent2 in env.agent_positions:
                            dist = np.linalg.norm(env.agent_positions[agent1] - env.agent_positions[agent2])
                            distances.append(float(dist))
            episode_data["distances_between_agents"].append(distances)
            
            # Update termination status
            for agent in env.agent_names:
                episode_data["terminated"][agent] = episode_data["terminated"][agent] or terminated.get(agent, False)
                episode_data["truncated"][agent] = episode_data["truncated"][agent] or truncated.get(agent, False)
            
            # Print step information
            if step % 10 == 0:
                print(f"Step {step}:")
                print(f"  Active agents: {env.agents}")
                print(f"  Target position: {env.target_position}")
                
                # Print rewards (first objective only if multi-objective)
                reward_str = "  Rewards: {"
                for agent in env.agent_names:
                    if agent in rewards:
                        if isinstance(rewards[agent], np.ndarray):
                            reward_str += f"'{agent}': {rewards[agent][0]:.4f}, "
                        else:
                            reward_str += f"'{agent}': {rewards[agent]:.4f}, "
                reward_str = reward_str[:-2] + "}"
                print(reward_str)
                
                # Print terminations and truncations
                print(f"  Terminations: {terminated}")
                print(f"  Truncations: {truncated}")
                print()
            
            step += 1
        
        # Store final status
        episode_data["final_step"] = step
        episode_data["all_terminated"] = all(episode_data["terminated"].values())
        episode_data["all_truncated"] = all(episode_data["truncated"].values())
        
        all_episode_data.append(episode_data)
    
    return all_episode_data


def generate_visualizations(episode_data: List[Dict[str, Any]], output_dir: str):
    """
    Generate visualization assets from episode data.
    
    Args:
        episode_data: List of episode data dictionaries
        output_dir: Directory to save visualization assets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data as JSON
    with open(os.path.join(output_dir, "simulation_data.json"), "w") as f:
        json.dump(episode_data, f, indent=2)
    
    # Process the first episode for visualizations
    episode = episode_data[0]
    
    # 1. Plot trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot agent trajectories
    for agent, positions in episode["positions"].items():
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=agent)
        
        # Plot start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], marker="o", s=50)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], marker="x", s=50)
    
    # Plot target trajectory
    target_positions = np.array(episode["target_positions"])
    ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], "k--", label="Target")
    ax.scatter(target_positions[0, 0], target_positions[0, 1], target_positions[0, 2], marker="o", s=50, c="k")
    ax.scatter(target_positions[-1, 0], target_positions[-1, 1], target_positions[-1, 2], marker="x", s=50, c="k")
    
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Agent and Target Trajectories")
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectories.png"), dpi=300)
    plt.close()
    
    # 2. Plot rewards over time
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if rewards are multi-objective
    is_multi_obj = isinstance(episode["rewards"][episode["rewards"].keys().__iter__().__next__()][0], list)
    
    if is_multi_obj:
        # Plot first objective (close to target)
        for agent, rewards in episode["rewards"].items():
            rewards_obj1 = [r[0] for r in rewards]
            ax.plot(rewards_obj1, label=f"{agent} (close to target)")
        
        # Create a second plot for the second objective
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for agent, rewards in episode["rewards"].items():
            rewards_obj2 = [r[1] for r in rewards]
            ax2.plot(rewards_obj2, label=f"{agent} (far from others)")
        
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Reward (far from others)")
        ax2.set_title("Rewards Over Time (Objective 2: Far from Others)")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rewards_obj2.png"), dpi=300)
        plt.close(fig2)
    else:
        # Plot scalar rewards
        for agent, rewards in episode["rewards"].items():
            ax.plot(rewards, label=agent)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (close to target)")
    ax.set_title("Rewards Over Time (Objective 1: Close to Target)")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rewards.png"), dpi=300)
    plt.close()
    
    # 3. Plot distances to target
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for agent, distances in episode["distances_to_target"].items():
        ax.plot(distances, label=agent)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to Target")
    ax.set_title("Agent Distances to Target Over Time")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distances.png"), dpi=300)
    plt.close()
    
    # 4. Create animation of the simulation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Set axis limits
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 3])
    
    # Initialize plots
    agent_plots = {}
    for agent in episode["positions"]:
        agent_plots[agent] = ax.plot([], [], [], "o-", label=agent)[0]
    
    target_plot = ax.plot([], [], [], "ro", markersize=10, label="Target")[0]
    
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Catch Environment Simulation")
    ax.legend()
    
    def init():
        for agent in agent_plots:
            agent_plots[agent].set_data([], [])
            agent_plots[agent].set_3d_properties([])
        
        target_plot.set_data([], [])
        target_plot.set_3d_properties([])
        
        return list(agent_plots.values()) + [target_plot]
    
    def animate(i):
        # Update agent positions
        for agent in agent_plots:
            positions = np.array(episode["positions"][agent])
            x = positions[max(0, i-10):i+1, 0]  # Show trail of last 10 steps
            y = positions[max(0, i-10):i+1, 1]
            z = positions[max(0, i-10):i+1, 2]
            agent_plots[agent].set_data(x, y)
            agent_plots[agent].set_3d_properties(z)
        
        # Update target position
        target_positions = np.array(episode["target_positions"])
        target_plot.set_data([target_positions[i, 0]], [target_positions[i, 1]])
        target_plot.set_3d_properties([target_positions[i, 2]])
        
        return list(agent_plots.values()) + [target_plot]
    
    # Create animation
    num_frames = min(len(episode["target_positions"]), 100)  # Limit to 100 frames for efficiency
    step_size = max(1, len(episode["target_positions"]) // num_frames)
    frames = range(0, len(episode["target_positions"]), step_size)
    
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames,
        interval=100, blit=True
    )
    
    # Save animation
    anim.save(os.path.join(output_dir, "simulation.gif"), writer="pillow", fps=10)
    plt.close()
    
    # 5. Generate a simulation report
    report = f"""# Catch Environment Simulation Report

## Overview

This report summarizes the results of running a simulation of the Catch environment with {NUM_AGENTS} agents using {NUM_POLICIES} different policies.

## Simulation Parameters

- Number of agents: {NUM_AGENTS}
- Number of policies: {NUM_POLICIES}
- Number of episodes: {NUM_EPISODES}
- Maximum steps per episode: {MAX_STEPS}
- Random seed: {RANDOM_SEED}
- Multi-objective rewards: {episode_data[0]["rewards"][list(episode_data[0]["rewards"].keys())[0]][0].__class__.__name__ == "list"}

## Results Summary

### Episode 1

- Total steps: {episode["final_step"]}
- All agents terminated: {episode["all_terminated"]}
- All agents truncated: {episode["all_truncated"]}

### Agent Performance

| Agent | Avg. Distance to Target | Final Distance to Target |
|-------|-------------------------|--------------------------|
"""
    
    for agent in episode["distances_to_target"]:
        avg_distance = np.mean(episode["distances_to_target"][agent])
        final_distance = episode["distances_to_target"][agent][-1]
        report += f"| {agent} | {avg_distance:.4f} | {final_distance:.4f} |\n"
    
    report += """
## Visualization Assets

The following visualization assets were generated:

1. `trajectories.png`: 3D visualization of agent and target trajectories
2. `rewards.png`: Plot of rewards over time for each agent
3. `distances.png`: Plot of distances to target over time for each agent
4. `simulation.gif`: Animation of the simulation
5. `simulation_data.json`: Raw simulation data in JSON format

## Multi-Objective Analysis

The simulation demonstrates the trade-offs between different objectives:

1. **Close to target**: Agents try to minimize their distance to the target
2. **Far from others**: Agents try to maintain distance from other agents to avoid collisions

Different policies prioritize these objectives differently, resulting in different behaviors.
"""
    
    # Save report
    with open(os.path.join(output_dir, "simulation_report.md"), "w") as f:
        f.write(report)


def main():
    """Main function to run the simulation and generate visualizations."""
    print("Running Catch environment simulation with multi-objective reinforcement learning...")
    
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
    
    # Run simulation
    episode_data = run_simulation(
        env=env,
        policies=policies,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        render=RENDER,
    )
    
    # Generate visualizations
    generate_visualizations(episode_data, RESULTS_DIR)
    
    print(f"Results saved to {RESULTS_DIR}")
    print("Simulation complete!")


if __name__ == "__main__":
    main()