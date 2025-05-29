"""Training script for Multi-Objective MAPPO on the Catch environment."""
import argparse
import os
import time
from distutils.util import strtobool

import numpy as np
import jax
import jax.numpy as jnp

from learning.fulljax.momappo_fulljax import make_train, equally_spaced_weights
from crazy_rl.multi_agent.jax.catch import Catch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="mo_catch",
                        help="name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--num-weights", type=int, default=10,
                        help="number of weight vectors to generate")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-speed", type=float, default=0.15,
                        help="speed of the target")
    parser.add_argument("--num-agents", type=int, default=4,
                        help="number of agents")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="run in debug mode with reduced parameters")
    parser.add_argument("--activation", type=str, default="tanh",
                        help="activation function for neural networks")
    return parser.parse_args()


def main():
    """Main function to train multi-objective policies."""
    args = parse_args()
    
    # Reduce parameters for debug mode
    if args.debug:
        args.num_weights = 3
        args.total_timesteps = 10000
        args.num_envs = 2
        args.num_steps = 64
    
    # Create results directory if it doesn't exist
    os.makedirs(f"results/{args.exp_name}/models", exist_ok=True)
    
    # Initialize environment function
    def env_fn():
        return Catch(
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
            multi_obj=True,  # Enable multi-objective rewards
            size=5,
        )
    
    print(f"Training {args.num_weights} policies with different objective weightings")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Target speed: {args.target_speed}")
    
    # Generate weights for multi-objective optimization
    weights = jnp.array(equally_spaced_weights(2, args.num_weights))
    print(f"Generated {args.num_weights} weight vectors: {weights}")
    
    # Initialize random key
    rng = jax.random.PRNGKey(args.seed)
    start_time = time.time()
    
    # Create and jit the training function
    train_vjit = jax.jit(
        jax.vmap(make_train(args), in_axes=(None, 0)),  # vmaps over the weights
    )
    
    # Train multiple policies in parallel
    print("Starting training...")
    out = jax.block_until_ready(train_vjit(rng, weights))
    
    # Print training statistics
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Steps per second: {args.total_timesteps * args.num_weights / training_time:.2f}")
    
    # Save models
    for i in range(args.num_weights):
        actor_i = jax.tree_map(lambda x: x[i], out["runner_state"][0])
        model_path = f"results/{args.exp_name}/models/policy_{i}"
        os.makedirs(model_path, exist_ok=True)
        print(f"Saving model {i} to {model_path}")
    
    print(f"Training complete. Models saved to results/{args.exp_name}/models/")
    print("To visualize policies, run:")
    print(f"  python visualize_catch.py --model-path results/{args.exp_name}/models/policy_0")
    print("To generate Pareto front, run:")
    print(f"  python construct_pareto_front_catch.py --models-dir results/{args.exp_name}/models")


if __name__ == "__main__":
    main()