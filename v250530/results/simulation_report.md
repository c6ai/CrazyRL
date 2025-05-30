# Catch Environment Simulation Report

## Overview

This report summarizes the results of running a simulation of the Catch environment with 4 agents using 5 different policies.

## Simulation Parameters

- Number of agents: 4
- Number of policies: 5
- Number of episodes: 3
- Maximum steps per episode: 100
- Random seed: 42
- Multi-objective rewards: True

## Results Summary

### Episode 1

- Total steps: 28
- All agents terminated: True
- All agents truncated: False

### Agent Performance

| Agent | Avg. Distance to Target | Final Distance to Target |
|-------|-------------------------|--------------------------|
| agent_0 | 1.5037 | 1.1488 |
| agent_1 | 1.1492 | 1.0369 |
| agent_2 | 2.5793 | 3.1106 |
| agent_3 | 3.8398 | 4.4583 |

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
