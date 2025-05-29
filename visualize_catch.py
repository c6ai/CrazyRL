"""Visualization script for trained policies on the Catch environment."""
import argparse
import time
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import orbax.checkpoint
from flax.training.train_state import TrainState
from etils import epath

from crazy_rl.multi_agent.numpy.catch import Catch


class Actor(nn.Module):
    """Actor network for MAPPO."""
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, local_obs_and_id: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), 
                             bias_init=nn.initializers.constant(0.0))(local_obs_and_id)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), 
                             bias_init=nn.initializers.constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), 
                             bias_init=nn.initializers.constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(actor_logtstd)
        pi = distrax.MultivariateNormalDiag(actor_mean, std)
        return pi


def _one_hot(agent_id: int, num_agents: int):
    """Create one-hot encoding of agent ID."""
    return jnp.eye(num_agents)[agent_id]


def _norm_obs(obs: np.ndarray, min_obs: float, max_obs: float, low: float = -1.0, high: float = 1.0):
    """Normalize observations to [-1, 1] range."""
    return low + (obs - min_obs) * (high - low) / (max_obs - min_obs)


def load_policy(model_path: str):
    """Load trained policy from checkpoint."""
    print(f"Loading policy from {model_path}")
    directory = epath.Path(model_path)
    
    # Create dummy actor state to restore
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros(15)  # Adjust size based on observation space
    action_dim = 3  # 3D action space for drones
    
    actor_module = Actor(action_dim=action_dim)
    actor_state = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(key, dummy_obs),
        tx=jax.tree_util.Partial(lambda x: x),  # Dummy optimizer
    )
    
    # Load checkpoint
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    actor_state = ckptr.restore(model_path, item=actor_state)
    
    return actor_module, actor_state


def get_action(actor_module: Actor, actor_state: TrainState, env: Catch, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Get actions from policy for all agents."""
    actions = {}
    for i, (key, value) in enumerate(obs.items()):
        # Normalize observations
        normalized_obs = _norm_obs(
            value, 
            min(env.observation_space(key).low), 
            max(env.observation_space(key).high)
        )
        
        # Add agent ID to observations
        agent_id = _one_hot(i, len(obs))
        agent_obs = jnp.concatenate([jnp.asarray(normalized_obs), agent_id])
        
        # Get action from policy
        pi = actor_module.apply(actor_state.params, agent_obs)
        action = pi.mode()  # Deterministic action (mean)
        
        # Clip action to valid range
        action = jnp.clip(action, -1.0, 1.0)
        actions[key] = np.array(action)
    
    return actions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="path to the trained model")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--num-agents", type=int, default=4,
                        help="number of agents")
    parser.add_argument("--target-speed", type=float, default=0.15,
                        help="speed of the target")
    parser.add_argument("--render-mode", type=str, default="human",
                        help="render mode: 'human' for visualization, None for headless")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="maximum number of steps to run")
    return parser.parse_args()


def main():
    """Main function to visualize trained policy."""
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
    step = 0
    
    print("\n" + "="*50)
    print("INTRUDER EVASION SIMULATION")
    print("="*50)
    print("Target (ownship) is actively evading the intruders (drones)")
    print(f"Using policy from: {args.model_path}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Target speed: {args.target_speed}")
    print("="*50 + "\n")
    
    try:
        while not done and step < args.max_steps:
            # Get actions from policy
            actions = get_action(actor_module, actor_state, env, obs)
            
            # Step environment
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            
            # Check if done
            terminated = any(terminations.values())
            truncated = any(truncations.values())
            done = terminated or truncated
            
            # Print status
            if step % 10 == 0:  # Print every 10 steps to reduce output
                print(f"\nStep {step}:")
                print(f"  Target position: {next_obs[list(next_obs.keys())[0]][3:6]}")
                
                # Calculate average distance to target
                avg_dist = 0
                for agent, agent_obs in next_obs.items():
                    agent_pos = agent_obs[:3]
                    target_pos = agent_obs[3:6]
                    dist = np.linalg.norm(agent_pos - target_pos)
                    avg_dist += dist
                    print(f"  {agent} position: {agent_pos}, distance to target: {dist:.2f}")
                
                avg_dist /= len(next_obs)
                print(f"  Average distance to target: {avg_dist:.2f}")
                
                # Print rewards
                print(f"  Rewards: {rewards}")
            
            obs = next_obs
            step += 1
            
            # Slow down simulation for better visualization
            if args.render_mode == "human":
                time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        env.close()
        
    if done:
        if terminated:
            print("\nSimulation terminated: collision detected")
        else:
            print("\nSimulation truncated: maximum steps reached")
    
    print(f"Completed {step} steps")
    print("Simulation complete")


if __name__ == "__main__":
    main()