"""
Simulate the Catch environment and generate visualization assets.
This script creates a simplified simulation of the Catch environment
without requiring the actual crazy_rl package.
"""
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Create results directory
os.makedirs("/workspace/CrazyRL/v250529/results", exist_ok=True)

class SimpleCatchEnv:
    """A simplified version of the Catch environment."""
    
    def __init__(self, num_agents=4, target_speed=0.1):
        self.num_agents = num_agents
        self.target_speed = target_speed
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        
        # Initialize positions
        self.agent_positions = {
            "agent_0": np.array([0.0, 0.0, 1.0]),
            "agent_1": np.array([1.0, 1.0, 1.0]),
            "agent_2": np.array([0.0, 1.0, 1.0]),
            "agent_3": np.array([2.0, 2.0, 1.0])
        }
        self.target_position = np.array([1.0, 1.0, 2.5])
        
        # Environment bounds
        self.bounds = {
            "x": (-1.0, 3.0),
            "y": (-1.0, 3.0),
            "z": (0.0, 4.0)
        }
        
        # Termination conditions
        self.min_distance_to_target = 0.3  # Minimum distance to catch target
        self.min_distance_between_agents = 0.5  # Minimum distance between agents
        self.max_steps = 20
        self.current_step = 0
        
        # Rewards
        self.catch_reward = 10.0
        self.collision_penalty = -10.0
        self.out_of_bounds_penalty = -10.0
        self.step_penalty = -0.1
        
        # Termination flags
        self.terminated = {agent: False for agent in self.agents}
        self.truncated = {agent: False for agent in self.agents}
    
    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset positions
        self.agent_positions = {
            "agent_0": np.array([0.0, 0.0, 1.0]),
            "agent_1": np.array([1.0, 1.0, 1.0]),
            "agent_2": np.array([0.0, 1.0, 1.0]),
            "agent_3": np.array([2.0, 2.0, 1.0])
        }
        self.target_position = np.array([1.0, 1.0, 2.5])
        
        # Reset termination flags
        self.terminated = {agent: False for agent in self.agents}
        self.truncated = {agent: False for agent in self.agents}
        
        # Reset step counter
        self.current_step = 0
        
        # Return observations
        observations = {}
        for agent in self.agents:
            observations[agent] = np.concatenate([
                self.agent_positions[agent],
                self.target_position,
                np.zeros(9)  # Padding to match the original observation space
            ])
        
        return observations, {}
    
    def step(self, actions):
        """Take a step in the environment."""
        self.current_step += 1
        
        # Move agents based on actions
        for agent, action in actions.items():
            if not self.terminated[agent] and not self.truncated[agent]:
                # Scale action to reasonable movement
                scaled_action = np.clip(action, -1.0, 1.0) * 0.2
                
                # Update position
                self.agent_positions[agent] += scaled_action
                
                # Check bounds
                for i, (dim, (min_val, max_val)) in enumerate(zip(['x', 'y', 'z'], [
                    self.bounds['x'], self.bounds['y'], self.bounds['z']
                ])):
                    if self.agent_positions[agent][i] < min_val or self.agent_positions[agent][i] > max_val:
                        self.terminated[agent] = True
        
        # Move target (simple random walk)
        target_direction = np.random.uniform(-1, 1, 3)
        target_direction = target_direction / np.linalg.norm(target_direction)
        self.target_position += target_direction * self.target_speed
        
        # Keep target within bounds
        for i, (min_val, max_val) in enumerate([
            self.bounds['x'], self.bounds['y'], self.bounds['z']
        ]):
            self.target_position[i] = np.clip(self.target_position[i], min_val, max_val)
        
        # Calculate rewards and check termination conditions
        rewards = {}
        observations = {}
        
        active_agents = [agent for agent in self.agents if not self.terminated[agent] and not self.truncated[agent]]
        
        for agent in self.agents:
            if self.terminated[agent] or self.truncated[agent]:
                rewards[agent] = 0.0
                continue
            
            # Calculate distance to target
            distance_to_target = np.linalg.norm(self.agent_positions[agent] - self.target_position)
            
            # Check if agent caught the target
            if distance_to_target < self.min_distance_to_target:
                rewards[agent] = self.catch_reward
                self.terminated[agent] = True
            else:
                rewards[agent] = self.step_penalty - distance_to_target * 0.1
            
            # Check for collisions with other agents
            for other_agent in active_agents:
                if agent != other_agent:
                    distance = np.linalg.norm(self.agent_positions[agent] - self.agent_positions[other_agent])
                    if distance < self.min_distance_between_agents:
                        rewards[agent] = self.collision_penalty
                        self.terminated[agent] = True
                        break
            
            # Check for out of bounds
            for i, (min_val, max_val) in enumerate([
                self.bounds['x'], self.bounds['y'], self.bounds['z']
            ]):
                if self.agent_positions[agent][i] < min_val or self.agent_positions[agent][i] > max_val:
                    rewards[agent] = self.out_of_bounds_penalty
                    self.terminated[agent] = True
                    break
        
        # Check for truncation (max steps)
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                if not self.terminated[agent]:
                    self.truncated[agent] = True
        
        # Create observations
        for agent in self.agents:
            observations[agent] = np.concatenate([
                self.agent_positions[agent],
                self.target_position,
                np.zeros(9)  # Padding to match the original observation space
            ])
        
        # Update active agents list
        self.agents = active_agents
        
        return observations, rewards, self.terminated, self.truncated, {}
    
    def close(self):
        """Close the environment."""
        pass

def run_simulation(max_steps=15, seed=42):
    """Run the simulation and collect data."""
    np.random.seed(seed)
    
    # Initialize environment
    env = SimpleCatchEnv(num_agents=4, target_speed=0.1)
    
    # Reset environment
    observations, _ = env.reset(seed=seed)
    
    # Data collection
    simulation_data = {
        "steps": [],
        "rewards": [],
        "positions": [],
        "target_positions": [],
        "terminations": [],
        "truncations": []
    }
    
    # Run for max_steps or until all agents terminate
    step = 0
    while step < max_steps:
        if not env.agents:
            print(f"All agents terminated at step {step}")
            break
        
        # Use random actions (in a real scenario, you would use a trained policy)
        actions = {
            agent: np.random.uniform(-1, 1, 3) for agent in env.agents
        }
        
        # Step the environment
        observations, rewards, terminations, truncations, _ = env.step(actions)
        
        # Extract positions from observations
        positions = {}
        for agent, obs in observations.items():
            positions[agent] = obs[:3].tolist()  # Agent position
        
        # Store data
        simulation_data["steps"].append(step)
        simulation_data["rewards"].append({agent: float(reward) for agent, reward in rewards.items()})
        simulation_data["positions"].append(positions)
        simulation_data["target_positions"].append(env.target_position.tolist())
        simulation_data["terminations"].append({agent: bool(term) for agent, term in terminations.items()})
        simulation_data["truncations"].append({agent: bool(trunc) for agent, trunc in truncations.items()})
        
        # Print step information
        print(f"Step {step}:")
        print(f"  Agents: {env.agents}")
        print(f"  Target position: {env.target_position}")
        print(f"  Rewards: {rewards}")
        print(f"  Terminations: {terminations}")
        print(f"  Truncations: {truncations}")
        print("")
        
        step += 1
    
    # Close environment
    env.close()
    
    return simulation_data

def save_results(simulation_data, output_dir="/workspace/CrazyRL/v250529/results"):
    """Save simulation results to files."""
    # Save raw data as JSON
    with open(f"{output_dir}/simulation_data.json", "w") as f:
        json.dump(simulation_data, f, indent=2)
    
    # Create static visualizations
    create_trajectory_plot(simulation_data, output_dir)
    create_reward_plot(simulation_data, output_dir)
    create_distance_plot(simulation_data, output_dir)
    
    # Create animated visualization
    create_animation(simulation_data, output_dir)
    
    print(f"Results saved to {output_dir}")

def create_trajectory_plot(data, output_dir):
    """Create a 3D plot of agent and target trajectories."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot agent trajectories
    agent_colors = {
        'agent_0': 'blue',
        'agent_1': 'green',
        'agent_2': 'red',
        'agent_3': 'purple'
    }
    
    for agent in agent_colors:
        x, y, z = [], [], []
        for step in range(len(data["positions"])):
            if agent in data["positions"][step]:
                pos = data["positions"][step][agent]
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
        
        ax.plot(x, y, z, color=agent_colors[agent], label=agent)
        
        # Plot final position with a marker
        if x and y and z:
            ax.scatter(x[-1], y[-1], z[-1], color=agent_colors[agent], s=100, marker='o')
    
    # Plot target trajectory
    x, y, z = [], [], []
    for target_pos in data["target_positions"]:
        x.append(target_pos[0])
        y.append(target_pos[1])
        z.append(target_pos[2])
    
    ax.plot(x, y, z, color='black', linestyle='--', label='Target')
    ax.scatter(x[-1], y[-1], z[-1], color='black', s=100, marker='*')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Agent and Target Trajectories')
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectories.png", dpi=300)
    plt.close()

def create_reward_plot(data, output_dir):
    """Create a plot of rewards over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract rewards for each agent
    agent_rewards = {}
    for agent in data["rewards"][0].keys():
        agent_rewards[agent] = [step_rewards.get(agent, 0) for step_rewards in data["rewards"]]
    
    # Plot rewards
    for agent, rewards in agent_rewards.items():
        ax.plot(data["steps"][:len(rewards)], rewards, label=agent)
    
    # Set labels and title
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Agent Rewards Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rewards.png", dpi=300)
    plt.close()

def create_distance_plot(data, output_dir):
    """Create a plot of distances to target over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate distances for each agent
    agent_distances = {}
    for agent in data["positions"][0].keys():
        distances = []
        for step in range(len(data["positions"])):
            if agent in data["positions"][step]:
                agent_pos = np.array(data["positions"][step][agent])
                target_pos = np.array(data["target_positions"][step])
                distance = np.linalg.norm(agent_pos - target_pos)
                distances.append(distance)
        agent_distances[agent] = distances
    
    # Plot distances
    for agent, distances in agent_distances.items():
        ax.plot(data["steps"][:len(distances)], distances, label=agent)
    
    # Set labels and title
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Target')
    ax.set_title('Agent Distances to Target Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distances.png", dpi=300)
    plt.close()

def create_animation(data, output_dir):
    """Create an animated visualization of the simulation."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    x_min, x_max = -1, 3
    y_min, y_max = -1, 3
    z_min, z_max = 0, 4
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Catch Environment Simulation')
    
    # Initialize plots
    agent_plots = {}
    agent_colors = {
        'agent_0': 'blue',
        'agent_1': 'green',
        'agent_2': 'red',
        'agent_3': 'purple'
    }
    
    for agent in agent_colors:
        agent_plots[agent] = ax.plot([], [], [], 'o-', color=agent_colors[agent], label=agent)[0]
    
    target_plot = ax.plot([], [], [], '*-', color='black', label='Target')[0]
    
    # Add a text annotation for step count
    step_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    
    def init():
        for agent in agent_plots:
            agent_plots[agent].set_data([], [])
            agent_plots[agent].set_3d_properties([])
        
        target_plot.set_data([], [])
        target_plot.set_3d_properties([])
        step_text.set_text("")
        
        return list(agent_plots.values()) + [target_plot, step_text]
    
    def update(frame):
        # Update agent positions
        for agent in agent_plots:
            x, y, z = [], [], []
            for step in range(frame + 1):
                if step < len(data["positions"]) and agent in data["positions"][step]:
                    pos = data["positions"][step][agent]
                    x.append(pos[0])
                    y.append(pos[1])
                    z.append(pos[2])
            
            agent_plots[agent].set_data(x, y)
            agent_plots[agent].set_3d_properties(z)
        
        # Update target position
        x, y, z = [], [], []
        for step in range(frame + 1):
            if step < len(data["target_positions"]):
                pos = data["target_positions"][step]
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
        
        target_plot.set_data(x, y)
        target_plot.set_3d_properties(z)
        
        # Update step text
        if frame < len(data["steps"]):
            step_text.set_text(f"Step: {data['steps'][frame]}")
        
        return list(agent_plots.values()) + [target_plot, step_text]
    
    # Create animation
    num_frames = len(data["steps"])
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    
    # Save animation
    anim.save(f"{output_dir}/simulation.gif", writer='pillow', fps=5)
    plt.close()

def create_summary_report(data, output_dir):
    """Create a summary report of the simulation."""
    # Calculate statistics
    num_steps = len(data["steps"])
    
    # Calculate average rewards
    avg_rewards = {}
    for agent in data["rewards"][0].keys():
        rewards = [step_rewards.get(agent, 0) for step_rewards in data["rewards"]]
        avg_rewards[agent] = sum(rewards) / len(rewards)
    
    # Calculate average distances
    avg_distances = {}
    for agent in data["positions"][0].keys():
        distances = []
        for step in range(len(data["positions"])):
            if agent in data["positions"][step]:
                agent_pos = np.array(data["positions"][step][agent])
                target_pos = np.array(data["target_positions"][step])
                distance = np.linalg.norm(agent_pos - target_pos)
                distances.append(distance)
        avg_distances[agent] = sum(distances) / len(distances)
    
    # Check if any agent terminated
    terminated_agents = set()
    for step_terms in data["terminations"]:
        for agent, term in step_terms.items():
            if term:
                terminated_agents.add(agent)
    
    # Create report
    report = f"""# Catch Environment Simulation Report

## Summary
- Total steps: {num_steps}
- Terminated agents: {list(terminated_agents) if terminated_agents else "None"}

## Average Rewards
"""
    
    for agent, avg_reward in avg_rewards.items():
        report += f"- {agent}: {avg_reward:.4f}\n"
    
    report += "\n## Average Distances to Target\n"
    
    for agent, avg_distance in avg_distances.items():
        report += f"- {agent}: {avg_distance:.4f}\n"
    
    report += "\n## Final Positions\n"
    
    if data["positions"]:
        final_positions = data["positions"][-1]
        final_target = data["target_positions"][-1]
        
        report += f"- Target: [{final_target[0]:.2f}, {final_target[1]:.2f}, {final_target[2]:.2f}]\n"
        
        for agent, pos in final_positions.items():
            report += f"- {agent}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n"
    
    # Save report
    with open(f"{output_dir}/simulation_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    print("Running Catch environment simulation...")
    simulation_data = run_simulation(max_steps=13, seed=42)
    save_results(simulation_data)
    create_summary_report(simulation_data, "/workspace/CrazyRL/v250529/results")
    print("Simulation complete!")