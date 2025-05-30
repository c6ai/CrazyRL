# Multi-Objective Reinforcement Learning for Catch Environment (v250530)

This folder contains a self-contained implementation of the Catch environment with multi-objective reinforcement learning capabilities. The implementation focuses on creating realistic physics and smooth trajectories for intruders and evader craft.

## Contents

- `catch_env.py`: Self-contained implementation of the Catch environment
- `simulate_catch.py`: Script to run a simulation with multiple MOMARL-enabled agents
- `results/`: Directory containing visualization assets and simulation results

## Environment Description

The Catch environment simulates a scenario where multiple agents (drones) try to surround and catch a moving target (evader) while maintaining distance from each other to avoid collisions. This is a multi-objective reinforcement learning problem with two competing objectives:

1. **Close to target**: Agents try to minimize their distance to the target
2. **Far from others**: Agents try to maintain distance from other agents to avoid collisions

## Features

This implementation includes several enhancements over the previous version:

1. **Realistic Physics**: Agents and targets move with realistic physics, including:
   - Mass and inertia
   - Drag forces
   - Acceleration-based movement
   - Velocity limits

2. **Smooth Trajectories**: Both agents and targets move with smooth trajectories:
   - Continuous acceleration and velocity changes
   - Momentum preservation
   - Realistic turning behavior

3. **Multi-Objective Policies**: The simulation includes multiple policies with different objective weightings:
   - Some policies prioritize getting close to the target
   - Others prioritize staying away from other agents
   - The rest balance between these objectives

4. **Self-Contained Implementation**: The entire environment is implemented in a single file with no external dependencies beyond NumPy and Matplotlib.

## Running the Simulation

To run the simulation:

```bash
cd /path/to/CrazyRL/v250530
python simulate_catch.py
```

This will:
1. Create a Catch environment with 4 agents
2. Run a simulation with multiple policies
3. Generate visualization assets in the `results/` directory

## Visualization Assets

The simulation generates several visualization assets:

1. `trajectories.png`: 3D visualization of agent and target trajectories
2. `rewards.png`: Plot of rewards over time for each agent
3. `distances.png`: Plot of distances to target over time for each agent
4. `simulation.gif`: Animation of the simulation
5. `simulation_data.json`: Raw simulation data in JSON format
6. `simulation_report.md`: Summary report of the simulation results

## Multi-Objective Analysis

The simulation demonstrates the trade-offs between different objectives. By adjusting the weights in the `MOPolicy` class, you can create policies that prioritize different objectives, resulting in different behaviors:

- Policies that prioritize getting close to the target will be more aggressive in pursuit
- Policies that prioritize staying away from other agents will be more cautious
- Balanced policies will try to achieve both objectives

The Pareto front of these policies shows the trade-off between these competing objectives, where improving one objective typically comes at the cost of the other.