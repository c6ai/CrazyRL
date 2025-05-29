# Multi-Objective Reinforcement Learning for Intruder Evasion (IE) Simulation

This document outlines a comprehensive plan to:
1. Train multiple policies on the Catch environment using multi-objective reinforcement learning
2. Create a real-time visual simulation of intruders catching an evading target
3. Generate a Pareto front visualization to analyze policy performance across multiple objectives

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Training Multiple Policies](#training-multiple-policies)
- [Visual Simulation](#visual-simulation)
- [Pareto Front Analysis](#pareto-front-analysis)
- [Technical Implementation Details](#technical-implementation-details)

## Overview

The Catch environment simulates a scenario where multiple drones (intruders) attempt to surround and catch a moving target (ownship) that is actively trying to evade them. This is a multi-objective problem where agents need to balance:

1. **Close to target**: Minimizing distance to the target
2. **Far from others**: Maintaining safe distance from other agents
3. **Efficient**: Minimizing energy consumption (implicitly through movement optimization)

We'll train multiple policies with different objective weightings, visualize their performance in real-time, and analyze the trade-offs between objectives using Pareto front visualization.

## Environment Setup

### Prerequisites

```bash
# Clone the repository if not already done
git clone https://github.com/c6ai/CrazyRL.git
cd CrazyRL

# Install dependencies
poetry install
```

### Environment Configuration

The Catch environment is configured with the following parameters:

- **Intruders**: Multiple drones trying to catch the target
- **Ownship**: A moving target that actively evades the intruders
- **Objectives**:
  - Minimize distance to target
  - Maximize distance from other agents
  - Avoid collisions

## Training Multiple Policies

We'll use the Multi-Objective MAPPO (MOMAPPO) algorithm to train policies with different objective weightings.

### Step 1: Create Training Script

Create a script `train_mo_catch.py` that:

```python
import argparse
import numpy as np
import jax
from learning.fulljax.momappo_fulljax import train
from crazy_rl.multi_agent.jax.catch import Catch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="mo_catch")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-weights", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-speed", type=float, default=0.15)
    parser.add_argument("--num-agents", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize environment
    env_fn = lambda: Catch(
        drone_ids=np.arange(args.num_agents),
        init_flying_pos=np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 2.0, 2.0],
            [2.0, 0.5, 1.0],
            [2.0, 2.5, 2.0],
            [2.0, 1.0, 2.5],
            [0.5, 0.5, 0.5],
        ])[:args.num_agents],
        init_target_location=np.array([1.0, 1.0, 2.0]),
        target_speed=args.target_speed,
        multi_obj=True,
        size=5,
    )
    
    # Train with MOMAPPO
    train(
        env_fn=env_fn,
        num_weights=args.num_weights,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        exp_name=args.exp_name,
    )

if __name__ == "__main__":
    main()
```

### Step 2: Train Multiple Policies

Execute the training script to generate multiple policies with different objective weightings:

```bash
# Train with default parameters
python train_mo_catch.py --num-weights 10 --total-timesteps 2000000

# Train with more weights for a denser Pareto front
python train_mo_catch.py --num-weights 20 --total-timesteps 1000000 --exp-name mo_catch_dense
```

The training process will:
1. Generate multiple weight vectors for the objectives
2. Train policies in parallel using JAX's vmap
3. Save trained models to `results/mo_catch/models/`

## Visual Simulation

### Step 1: Create Visualization Script

Create a script `visualize_catch.py` that:

```python
import argparse
import numpy as np
import jax
import time
from crazy_rl.multi_agent.numpy.catch import Catch
from learning.execution.exec_mappo_policy import load_policy, get_action

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--target-speed", type=float, default=0.15)
    parser.add_argument("--render-mode", type=str, default="human")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize environment
    env = Catch(
        drone_ids=np.arange(args.num_agents),
        render_mode=args.render_mode,
        init_flying_pos=np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 2.0, 2.0],
            [2.0, 0.5, 1.0],
            [2.0, 2.5, 2.0],
            [2.0, 1.0, 2.5],
            [0.5, 0.5, 0.5],
        ])[:args.num_agents],
        init_target_location=np.array([1.0, 1.0, 2.0]),
        target_speed=args.target_speed,
    )
    
    # Load policy
    actor_module, actor_state = load_policy(args.model_path)
    
    # Run simulation
    obs, _ = env.reset(seed=args.seed)
    done = False
    
    print(f"Starting simulation with policy from {args.model_path}")
    print("Target (ownship) is trying to evade the intruders (drones)")
    
    while not done:
        # Get actions from policy
        actions = get_action(actor_module, actor_state, env, obs)
        
        # Step environment
        obs, rewards, terminations, truncations, _ = env.step(actions)
        
        # Check if done
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        done = terminated or truncated
        
        # Print status
        print(f"Rewards: {rewards}")
        print(f"Target position: {obs[list(obs.keys())[0]][3:6]}")
        
        time.sleep(0.05)  # Slow down simulation for better visualization
    
    env.close()
    print("Simulation complete")

if __name__ == "__main__":
    main()
```

### Step 2: Run Visual Simulation

Execute the visualization script with a trained policy:

```bash
# Run with a specific policy
python visualize_catch.py --model-path results/mo_catch/models/policy_0 --render-mode human

# Try different policies to see different behaviors
python visualize_catch.py --model-path results/mo_catch/models/policy_5 --render-mode human
```

## Pareto Front Analysis

### Step 1: Run Pareto Front Construction

Use the existing `construct_pareto_front.py` script to analyze the performance of different policies:

```bash
# Generate Pareto front visualization
python learning/execution/construct_pareto_front.py --models-dir results/mo_catch/models
```

This will:
1. Load all trained policies
2. Evaluate each policy on the environment
3. Plot the Pareto front showing the trade-off between objectives
4. Save the visualization to `results/mo/pareto_front.png`

### Step 2: Analyze Results

The Pareto front visualization will show:
- X-axis: "Close to target" objective performance
- Y-axis: "Far from others" objective performance
- Each point represents a different policy with different objective weightings

Policies on the Pareto front represent optimal trade-offs between objectives. No policy can improve one objective without sacrificing another.

## Technical Implementation Details

### Multi-Objective Reinforcement Learning

The MOMAPPO algorithm extends MAPPO (Multi-Agent Proximal Policy Optimization) to handle multiple objectives:

1. **Scalarization**: Converts vector rewards to scalar rewards using weighted sums
2. **Weight Generation**: Uses reference direction methods to generate diverse weight vectors
3. **Parallel Training**: Trains multiple policies simultaneously using JAX's vectorization

### Environment Dynamics

The Catch environment implements:

1. **Target Movement**: The target actively evades agents by moving away from their center of mass
2. **Collision Detection**: Terminates episodes when agents collide with each other, the ground, or the target
3. **Multi-Objective Rewards**:
   - Reward for minimizing distance to target
   - Reward for maximizing distance from other agents
   - Penalty for collisions

### Policy Architecture

The policies use:

1. **Actor-Critic Architecture**: Separate networks for policy (actor) and value function (critic)
2. **Observation Processing**: Normalizes observations and adds agent ID
3. **Action Processing**: Outputs continuous actions in [-1, 1] range, scaled to physical movements

### Visualization

The visualization uses:

1. **Real-time Rendering**: Shows 3D positions of agents and target
2. **Pareto Front Plot**: Visualizes trade-offs between objectives
3. **Interactive Elements**: Allows hovering over points to see policy details

## Conclusion

This plan provides a comprehensive approach to training, visualizing, and analyzing multi-objective reinforcement learning policies for the intruder evasion scenario. By following these steps, you can:

1. Train multiple policies with different objective weightings
2. Visualize the behavior of these policies in real-time
3. Analyze the trade-offs between objectives using Pareto front visualization

The resulting insights can help understand the fundamental trade-offs in multi-agent coordination problems and inform the design of more effective drone swarm strategies.