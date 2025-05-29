"""Script to construct and visualize the Pareto front for trained policies on the Catch environment."""
import argparse
import random
import os
from typing import Sequence

import chex
import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint
from distrax import MultivariateNormalDiag
from etils import epath
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from mplcursors import cursor
from pettingzoo import ParallelEnv

from crazy_rl.multi_agent.numpy.catch import Catch
from crazy_rl.utils.pareto import ParetoArchive


# NN from MAPPO
class Actor(nn.Module):
    """Actor network for MAPPO."""
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, local_obs_and_id: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(local_obs_and_id)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(actor_logtstd)
        pi: MultivariateNormalDiag = distrax.MultivariateNormalDiag(actor_mean, std)
        return pi


def _one_hot(agent_id: int, num_agents: int):
    """Create one-hot encoding of agent ID."""
    return jnp.eye(num_agents)[agent_id]


def _norm_obs(obs: np.ndarray, min_obs: float, max_obs: float, low: float = -1.0, high: float = 1.0):
    """Normalize observations to [-1, 1] range."""
    return low + (obs - min_obs) * (high - low) / (max_obs - min_obs)


def _ma_get_action(actor: Actor, actor_state: TrainState, env: ParallelEnv, obs: dict, keys: chex.PRNGKey) -> dict:
    """Gets the action for all agents."""
    actions = {}
    for i, (key, value) in enumerate(obs.items()):
        # normalize obs
        normalized_obs = _norm_obs(value, min(env.observation_space(key).low), max(env.observation_space(key).high))
        # add agent id to obs
        agent_id = _one_hot(i, len(obs))
        agent_obs = jnp.concatenate([jnp.asarray(normalized_obs), agent_id])
        # get action from NN
        pi = actor.apply(actor_state.params, agent_obs)
        action = pi.mode()  # deterministic mode just takes the mean
        # clip action
        action = jnp.clip(action, -1.0, 1.0)
        actions[key] = np.array(action)
    return actions


def parse_args():
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--models-dir", type=str, required=True, 
                        help="the dir of the model to load")
    parser.add_argument("--num-agents", type=int, default=4,
                        help="number of agents")
    parser.add_argument("--target-speed", type=float, default=0.15,
                        help="speed of the target")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="number of episodes to evaluate each policy")
    parser.add_argument("--output-dir", type=str, default="results/mo",
                        help="directory to save output files")
    return parser.parse_args()


def play_episode(actor_module, actor_state, env, init_obs, key):
    """Play one episode.

    Args:
        actor_module: the actor network
        actor_state: the actor network parameters
        env: the environment
        init_obs: initial observations
        key: the random key
    """
    obs = init_obs
    done = False
    ep_return = np.zeros(2)  # Two objectives: close to target, far from others
    
    while not done:
        # Execute policy for each agent
        key, subkey = jax.random.split(key)
        action_keys = jax.random.split(subkey, env.num_agents)
        actions = _ma_get_action(actor_module, actor_state, env, obs, action_keys)

        next_obs, r, terminateds, truncateds, _ = env.step(actions)
        
        # Sum rewards across agents for each objective
        rewards_array = np.array(list(r.values()))
        ep_return += rewards_array.sum(axis=0)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        done = terminated or truncated
        obs = next_obs
        
    return ep_return


def load_actor_state(model_path, actor_state: TrainState):
    """Load actor state from checkpoint."""
    directory = epath.Path(model_path)
    print("Loading actor from ", directory)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    actor_state = ckptr.restore(model_path, item=actor_state)
    return actor_state


def replay_simu(args):
    """Replay the simulation for multiple episodes and evaluate policies.

    Args:
        args: the arguments from the command line
    """
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Initialize environment
    env = Catch(
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

    # Initialize actor
    _ = env.reset(seed=args.seed)
    single_action_space = env.action_space(env.unwrapped.agents[0])
    key, actor_key = jax.random.split(key, 2)
    init_local_state = jnp.asarray(env.observation_space(env.unwrapped.agents[0]).sample())
    init_local_state_and_id = jnp.append(init_local_state, _one_hot(0, env.num_agents))
    
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Create actor module and state
    actor_module = Actor(single_action_space.shape[0])
    actor_state = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_local_state_and_id),
        tx=optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=0.01, eps=1e-5),  # not used
        ),
    )

    # Get all model directories
    p = epath.Path(args.models_dir)
    model_dirs = [f for f in p.iterdir() if f.is_dir()]
    print(f"Found {len(model_dirs)} models in {args.models_dir}")
    
    # Initialize Pareto archive
    pareto_front = ParetoArchive()
    
    # Evaluate each policy
    for model_dir in model_dirs:
        print(f"Evaluating {model_dir.name}...")
        actor_state = load_actor_state(model_dir, actor_state)
        
        # Run multiple episodes for more reliable evaluation
        policy_eval = np.zeros(2)
        for episode in range(args.num_episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            episode_return = play_episode(actor_module, actor_state, env, obs, key)
            policy_eval += episode_return
            print(f"  Episode {episode+1}: {episode_return}")
        
        # Average over episodes
        policy_eval /= args.num_episodes
        print(f"  Average return: {policy_eval}")
        
        # Add to Pareto archive
        pareto_front.add(candidate=model_dir, evaluation=policy_eval)

    env.close()
    return pareto_front


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # List model directories
    p = epath.Path(args.models_dir)
    model_dirs = [f for f in p.iterdir() if f.is_dir()]
    print(f"Found {len(model_dirs)} models in {args.models_dir}")

    # Construct Pareto front
    print("\nConstructing Pareto front...")
    pf = replay_simu(args=args)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot all policies
    for i, (candidate, eval) in enumerate(zip(pf.individuals, pf.evaluations)):
        plt.scatter(eval[0], eval[1], s=100, label=candidate.name, alpha=0.9, 
                   c="#5CB5FF" if candidate in pf.pareto_front else "#FF5C5C")
        plt.annotate(candidate.name, (eval[0], eval[1]), fontsize=8)
    
    # Highlight Pareto front
    pareto_x = [pf.evaluations[pf.individuals.index(ind)][0] for ind in pf.pareto_front]
    pareto_y = [pf.evaluations[pf.individuals.index(ind)][1] for ind in pf.pareto_front]
    
    # Sort points for line plot
    pareto_points = sorted(zip(pareto_x, pareto_y))
    pareto_x = [p[0] for p in pareto_points]
    pareto_y = [p[1] for p in pareto_points]
    
    plt.plot(pareto_x, pareto_y, 'k--', alpha=0.7)

    # Add labels and grid
    plt.ylabel("Far from others", fontsize=16)
    plt.xlabel("Close to target", fontsize=16)
    plt.title("Pareto Front of Multi-Objective Policies", fontsize=18)
    plt.grid(alpha=0.25)
    
    # Add legend for Pareto front vs. dominated policies
    plt.scatter([], [], c="#5CB5FF", label="Pareto Optimal")
    plt.scatter([], [], c="#FF5C5C", label="Dominated")
    plt.legend(fontsize=12)
    
    # Save figures
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/pareto_front_catch.png", dpi=600)
    plt.savefig(f"{args.output_dir}/pareto_front_catch.pdf", dpi=600)
    
    # Print results
    print("\nPareto Front Policies:")
    for ind in pf.pareto_front:
        eval = pf.evaluations[pf.individuals.index(ind)]
        print(f"  {ind.name}: Close to target = {eval[0]:.4f}, Far from others = {eval[1]:.4f}")
    
    # Show interactive plot
    cursor(hover=True)
    plt.show()


if __name__ == "__main__":
    main()