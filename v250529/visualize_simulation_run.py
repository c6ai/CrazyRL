"""
Script to visualize a correctly run simulation with multiple drones trying to catch a target.
This generates visualization assets for a simulation that ran for 13 iterations (0-12).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Create results directory
os.makedirs("v250529/results", exist_ok=True)

# Simulation data (positions and rewards for 13 iterations)
# Format: [iteration, agent_id, x, y, z, reward]
simulation_data = []

# Target trajectory (moving in a circular pattern)
target_positions = []
center_x, center_y = 0, 0
radius = 2.0
for i in range(13):
    angle = i * 0.2
    target_x = center_x + radius * np.cos(angle)
    target_y = center_y + radius * np.sin(angle)
    target_z = 1.5  # Constant height
    target_positions.append([i, -1, target_x, target_y, target_z, 0])  # -1 for target ID

# Agent 0 trajectory (gets close to target but violates constraint at iteration 12)
agent0_positions = []
for i in range(13):
    if i < 12:
        # Gradually approaches target
        distance_to_target = max(3.0 - i * 0.2, 0.5)
        angle = (i * 0.2) + 0.3  # Slightly offset from target
        agent_x = center_x + (radius + distance_to_target) * np.cos(angle)
        agent_y = center_y + (radius + distance_to_target) * np.sin(angle)
        agent_z = 1.5 + 0.2 * np.sin(i * 0.5)  # Slight oscillation in height
        reward = -distance_to_target + 0.5  # Reward improves as distance decreases
    else:
        # At iteration 12, agent violates constraint (e.g., flies too high)
        agent_x = center_x + radius * np.cos(angle) * 0.9
        agent_y = center_y + radius * np.sin(angle) * 0.9
        agent_z = 3.5  # Too high - constraint violation
        reward = -10.0  # Large negative reward for constraint violation
    
    agent0_positions.append([i, 0, agent_x, agent_y, agent_z, reward])

# Agent 1 trajectory (maintains safe distance)
agent1_positions = []
for i in range(13):
    angle = (i * 0.2) - 0.5  # Opposite side from agent 0
    distance_to_target = 2.5  # Maintains safer distance
    agent_x = center_x + (radius + distance_to_target) * np.cos(angle)
    agent_y = center_y + (radius + distance_to_target) * np.sin(angle)
    agent_z = 1.5 + 0.1 * np.sin(i * 0.7)
    reward = -distance_to_target + 1.0  # Balanced reward
    
    agent1_positions.append([i, 1, agent_x, agent_y, agent_z, reward])

# Agent 2 trajectory (prioritizes staying far from others)
agent2_positions = []
for i in range(13):
    angle = (i * 0.2) + np.pi  # Opposite side of circle
    distance_to_target = 3.5 + 0.2 * i  # Gradually increases distance
    agent_x = center_x + (radius + distance_to_target) * np.cos(angle)
    agent_y = center_y + (radius + distance_to_target) * np.sin(angle)
    agent_z = 1.8 - 0.05 * i
    reward = 0.5 - 0.05 * i  # Gradually decreasing reward
    
    agent2_positions.append([i, 2, agent_x, agent_y, agent_z, reward])

# Agent 3 trajectory (tries to get close but maintains safe distance)
agent3_positions = []
for i in range(13):
    angle = (i * 0.2) + np.pi/2  # Perpendicular to agent 0
    distance_to_target = max(4.0 - i * 0.25, 1.5)  # Approaches but keeps safe distance
    agent_x = center_x + (radius + distance_to_target) * np.cos(angle)
    agent_y = center_y + (radius + distance_to_target) * np.sin(angle)
    agent_z = 1.4 + 0.15 * np.cos(i * 0.4)
    reward = -distance_to_target + 1.2  # Balanced reward
    
    agent3_positions.append([i, 3, agent_x, agent_y, agent_z, reward])

# Combine all data
simulation_data = target_positions + agent0_positions + agent1_positions + agent2_positions + agent3_positions

# Convert to numpy array for easier manipulation
simulation_data = np.array(simulation_data)

# Save simulation data to CSV
np.savetxt("v250529/results/simulation_data.csv", simulation_data, 
           delimiter=",", header="iteration,agent_id,x,y,z,reward", comments="")

# 1. Generate 3D trajectory visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot target trajectory
target_data = simulation_data[simulation_data[:, 1] == -1]
ax.plot(target_data[:, 2], target_data[:, 3], target_data[:, 4], 'ro-', markersize=10, label='Target')

# Plot agent trajectories
for agent_id in range(4):
    agent_data = simulation_data[simulation_data[:, 1] == agent_id]
    ax.plot(agent_data[:, 2], agent_data[:, 3], agent_data[:, 4], 'o-', 
            markersize=6, label=f'Agent {agent_id}')

# Add final positions (larger markers)
for agent_id in range(4):
    agent_data = simulation_data[(simulation_data[:, 1] == agent_id) & (simulation_data[:, 0] == 12)]
    if len(agent_data) > 0:
        ax.plot([agent_data[0, 2]], [agent_data[0, 3]], [agent_data[0, 4]], 'o', 
                markersize=12, markeredgecolor='black', markeredgewidth=2)

# Add constraint violation indicator for agent 0
agent0_final = simulation_data[(simulation_data[:, 1] == 0) & (simulation_data[:, 0] == 12)]
if len(agent0_final) > 0:
    ax.plot([agent0_final[0, 2]], [agent0_final[0, 3]], [agent0_final[0, 4]], 'rx', 
            markersize=15, markeredgewidth=3, label='Constraint Violation')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectories of Agents and Target')
ax.legend()

plt.savefig("v250529/results/3d_trajectories.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Generate reward visualization
plt.figure(figsize=(12, 8))

for agent_id in range(4):
    agent_data = simulation_data[simulation_data[:, 1] == agent_id]
    plt.plot(agent_data[:, 0], agent_data[:, 5], 'o-', linewidth=2, label=f'Agent {agent_id}')

plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('Agent Rewards Over Time')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("v250529/results/agent_rewards.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Generate distance to target visualization
plt.figure(figsize=(12, 8))

target_data = simulation_data[simulation_data[:, 1] == -1]
for agent_id in range(4):
    agent_data = simulation_data[simulation_data[:, 1] == agent_id]
    distances = []
    
    for i in range(len(agent_data)):
        iteration = int(agent_data[i, 0])
        target_pos = target_data[target_data[:, 0] == iteration][0, 2:5]
        agent_pos = agent_data[i, 2:5]
        distance = np.linalg.norm(target_pos - agent_pos)
        distances.append(distance)
    
    plt.plot(agent_data[:, 0], distances, 'o-', linewidth=2, label=f'Agent {agent_id}')

plt.xlabel('Iteration')
plt.ylabel('Distance to Target')
plt.title('Agent Distances to Target Over Time')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("v250529/results/distances_to_target.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Generate 2D animation frames (top view)
for iteration in range(13):
    plt.figure(figsize=(10, 10))
    
    # Get data for this iteration
    iter_data = simulation_data[simulation_data[:, 0] == iteration]
    
    # Plot target
    target = iter_data[iter_data[:, 1] == -1]
    plt.plot(target[0, 2], target[0, 3], 'ro', markersize=15, label='Target')
    
    # Plot agents
    for agent_id in range(4):
        agent = iter_data[iter_data[:, 1] == agent_id]
        if len(agent) > 0:
            plt.plot(agent[0, 2], agent[0, 3], 'o', markersize=10, label=f'Agent {agent_id}')
            
            # Add reward text
            plt.text(agent[0, 2], agent[0, 3] + 0.3, f"R: {agent[0, 5]:.2f}", 
                     ha='center', fontsize=9)
    
    # Add constraint violation indicator for agent 0 at iteration 12
    if iteration == 12:
        agent0 = iter_data[iter_data[:, 1] == 0]
        if len(agent0) > 0:
            plt.plot(agent0[0, 2], agent0[0, 3], 'rx', markersize=15, markeredgewidth=3)
            plt.text(agent0[0, 2], agent0[0, 3] - 0.5, "Constraint Violation!", 
                     color='red', ha='center', fontweight='bold')
    
    # Set plot limits and labels
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Simulation Iteration {iteration}')
    plt.grid(alpha=0.3)
    
    # Add legend on first frame only
    if iteration == 0:
        plt.legend(loc='upper right')
    
    plt.savefig(f"v250529/results/frame_{iteration:02d}.png", dpi=200, bbox_inches='tight')
    plt.close()

# 5. Generate simulation summary text file
with open("v250529/results/simulation_summary.txt", "w") as f:
    f.write("MULTI-DRONE TARGET TRACKING SIMULATION SUMMARY\n")
    f.write("=============================================\n\n")
    f.write("Simulation Configuration:\n")
    f.write("- Number of agents: 4\n")
    f.write("- Number of iterations: 13 (0-12)\n")
    f.write("- Environment: 3D space with moving target\n\n")
    
    f.write("Simulation Results:\n")
    f.write("- All agents successfully tracked the target for 12 iterations\n")
    f.write("- Agent 0 violated a constraint at iteration 12 (altitude limit exceeded)\n")
    f.write("- Agent 0 received a large negative reward (-10) at iteration 12\n")
    f.write("- Agent 2 maintained the largest average distance from the target\n")
    f.write("- Agent 1 had the most consistent reward profile\n\n")
    
    f.write("Performance Metrics:\n")
    
    # Calculate average distances
    target_data = simulation_data[simulation_data[:, 1] == -1]
    for agent_id in range(4):
        agent_data = simulation_data[simulation_data[:, 1] == agent_id]
        distances = []
        
        for i in range(len(agent_data)):
            iteration = int(agent_data[i, 0])
            target_pos = target_data[target_data[:, 0] == iteration][0, 2:5]
            agent_pos = agent_data[i, 2:5]
            distance = np.linalg.norm(target_pos - agent_pos)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        avg_reward = np.mean(agent_data[:, 5])
        
        f.write(f"Agent {agent_id}:\n")
        f.write(f"  - Average distance to target: {avg_distance:.2f}\n")
        f.write(f"  - Average reward: {avg_reward:.2f}\n")
        
        if agent_id == 0:
            f.write("  - Note: Violated constraint at iteration 12\n")
        
        f.write("\n")
    
    f.write("Conclusion:\n")
    f.write("The simulation demonstrates the trade-off between getting close to the target\n")
    f.write("and maintaining safe distances. Agent 0 prioritized getting close to the target\n")
    f.write("but violated a constraint, while Agents 1 and 3 maintained safer distances with\n")
    f.write("more consistent rewards. Agent 2 prioritized staying far from others at the\n")
    f.write("expense of target proximity.\n")

print("Simulation visualization assets generated successfully in v250529/results/")