"""Training script for Multi-Objective MAPPO on the Catch environment."""
import argparse
import os
import time
from distutils.util import strtobool

import numpy as np
import jax
import jax.numpy as jnp

from learning.fulljax.momappo_fulljax import equally_spaced_weights
from learning.fulljax.momappo_catch import make_train
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
    return parser.parse_args()


def train(
    env_fn,
    num_weights,
    seed,
    total_timesteps,
    learning_rate,
    num_envs,
    num_steps,
    update_epochs,
    num_minibatches,
    gamma,
    gae_lambda,
    clip_coef,
    ent_coef,
    vf_coef,
    max_grad_norm,
    exp_name,
    num_agents,
):
    """Train multiple policies with different objective weightings."""
    # Generate weights
    weights = equally_spaced_weights(num_weights, 3)  # 3 objectives
    
    # Create args object for make_train
    class Args:
        def __init__(self):
            self.env_fn = env_fn
            self.total_timesteps = total_timesteps
            self.lr = learning_rate
            self.num_envs = num_envs
            self.num_steps = num_steps
            self.update_epochs = update_epochs
            self.num_minibatches = num_minibatches
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_eps = clip_coef  # Note: clip_coef in our args is clip_eps in momappo
            self.ent_coef = ent_coef
            self.vf_coef = vf_coef
            self.max_grad_norm = max_grad_norm
            self.exp_name = exp_name
            self.seed = seed
            # Add default values for other required parameters
            self.activation = "tanh"
            self.anneal_lr = True
            self.debug = False
            self.num_agents = num_agents
    
    args = Args()
    
    # Create training function
    train_fn = make_train(args)
    
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    
    # Train with all weights
    train_fn(key, weights)


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
            num_drones=args.num_agents,
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
        num_agents=args.num_agents,
    )
    
    print(f"Training complete. Models saved to results/{args.exp_name}/models/")
    print("To visualize policies, run:")
    print(f"  python visualize_catch.py --model-path results/{args.exp_name}/models/policy_0")
    print("To generate Pareto front, run:")
    print(f"  python learning/execution/construct_pareto_front.py --models-dir results/{args.exp_name}/models")


if __name__ == "__main__":
    main()