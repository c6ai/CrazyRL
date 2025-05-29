# Multi-Objective Reinforcement Learning for Catch Environment

This folder contains the results of running a simulation of the Catch environment with multiple agents. The simulation was executed on May 29, 2025, as part of the multi-objective reinforcement learning project.

## Simulation Overview

The simulation ran successfully in headless mode. We can see the simulation running for 13 iterations (0-12) before completion. Here's what happened during the simulation:

- Four drones (agents 0-3) were trying to catch a moving target
- At each iteration, the agents received rewards based on their actions
- We can see the positions of each agent and the target at each step
- The simulation completed after 13 steps with all agents still active

## Generated Assets

The following visualization assets were generated:

1. **trajectories.png**: A 3D visualization of the paths taken by all agents and the target
2. **rewards.png**: A plot showing the rewards received by each agent over time
3. **distances.png**: A plot showing the distance of each agent to the target over time
4. **simulation.gif**: An animated visualization of the entire simulation
5. **simulation_data.json**: Raw data from the simulation in JSON format
6. **simulation_report.md**: A summary report of the simulation results

## Multi-Objective Analysis

The simulation demonstrates the trade-offs between different objectives:

1. **Close to target**: Agents try to minimize their distance to the target
2. **Far from others**: Agents try to maintain distance from other agents to avoid collisions

The Pareto front visualizations in the parent directory show the trade-off between these competing objectives. Different policies can be trained to prioritize one objective over the other, resulting in different behaviors.

## Next Steps

- Train actual policies with different objective weightings
- Generate a Pareto front based on actual trained policies
- Analyze the performance of different policies in the environment