"""Simple training script for the Catch environment."""
import os
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from crazy_rl.multi_agent.jax.catch import Catch
from crazy_rl.utils.jax_wrappers import (
    ClipActions,
    NormalizeObservation,
    LinearizeReward,
    AutoReset,
    VecEnv,
    NormalizeVecReward,
)
from learning.fulljax.momappo_fulljax import Actor, Critic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="v250529/mo_catch_simple",
                        help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total number of timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Number of steps per environment per update")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="Number of policy update epochs")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Number of minibatches")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm")
    parser.add_argument("--num-agents", type=int, default=4,
                        help="Number of agents")
    parser.add_argument("--target-speed", type=float, default=0.15,
                        help="Speed of the target")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.33, 0.33, 0.34],
                        help="Weights for the 3 objectives")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    # Create results directory
    os.makedirs(f"results/{args.exp_name}", exist_ok=True)
    
    # Create environment
    def create_env():
        env = Catch(
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
            multi_obj=True,
            size=5,
        )
        
        # Apply wrappers
        env = ClipActions(env)
        env = NormalizeObservation(env)
        env = LinearizeReward(env, jnp.array(args.weights))
        env = AutoReset(env)
        env = VecEnv(env)
        env = NormalizeVecReward(env, args.gamma)
        
        return env
    
    # Create environment
    env = create_env()
    
    # Initial reset to get observation shape
    key, subkey = jax.random.split(key)
    obs, info, state = env.reset(key=subkey)
    
    # Create actor and critic networks
    actor = Actor(env.action_space(0).shape[0], activation="tanh")
    critic = Critic(activation="tanh")
    
    # Initialize networks
    key, actor_key, critic_key = jax.random.split(key, 3)
    actor_params = actor.init(actor_key, jnp.zeros(env.observation_space(0).shape))
    critic_params = critic.init(critic_key, jnp.zeros(env.state(state).shape))
    
    # Create optimizers
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.learning_rate, eps=1e-5),
    )
    
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optimizer,
    )
    
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=optimizer,
    )
    
    # Training loop
    print(f"Starting training with weights: {args.weights}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Number of agents: {args.num_agents}")
    
    num_updates = args.total_timesteps // args.num_steps // args.num_envs
    
    # Simple training loop
    for update in range(num_updates):
        # Collect trajectories
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []
        log_prob_buffer = []
        
        # Reset at the beginning of each update
        if update == 0:
            key, subkey = jax.random.split(key)
            obs, info, state = env.reset(key=subkey)
        
        # Collect data
        for step in range(args.num_steps):
            # Get global state for critic
            global_obs = env.state(state)
            
            # Compute values
            values = critic.apply(critic_state.params, global_obs)
            
            # Sample actions for each agent
            actions = []
            log_probs = []
            
            for agent_id in range(args.num_agents):
                key, subkey = jax.random.split(key)
                dist = actor.apply(actor_state.params, obs[:, agent_id])
                action = dist.sample(seed=subkey)
                log_prob = dist.log_prob(action)
                actions.append(action)
                log_probs.append(log_prob)
            
            # Stack actions and log probs
            joint_actions = jnp.stack(actions, axis=1)
            log_probs = jnp.stack(log_probs, axis=1)
            
            # Store data
            obs_buffer.append(obs)
            action_buffer.append(joint_actions)
            value_buffer.append(values)
            log_prob_buffer.append(log_probs)
            
            # Step environment
            key, subkey = jax.random.split(key)
            keys_step = jax.random.split(subkey, args.num_envs)
            obs, rewards, terminateds, truncateds, info, state = env.step(state, joint_actions, jnp.stack(keys_step))
            
            # Store rewards and dones
            reward_buffer.append(rewards)
            done_buffer.append(terminateds | truncateds)
        
        # Convert buffers to arrays
        obs_buffer = jnp.stack(obs_buffer)
        action_buffer = jnp.stack(action_buffer)
        reward_buffer = jnp.stack(reward_buffer)
        done_buffer = jnp.stack(done_buffer)
        value_buffer = jnp.stack(value_buffer)
        log_prob_buffer = jnp.stack(log_prob_buffer)
        
        # Compute returns and advantages
        # Get last value
        last_value = critic.apply(critic_state.params, env.state(state))
        
        # Compute GAE
        advantages = jnp.zeros_like(reward_buffer)
        lastgaelam = 0
        
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - (done_buffer[-1])
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - done_buffer[t + 1]
                nextvalues = value_buffer[t + 1]
            
            delta = reward_buffer[t] + args.gamma * nextvalues * nextnonterminal - value_buffer[t]
            advantages = advantages.at[t].set(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
            lastgaelam = advantages[t]
        
        returns = advantages + value_buffer
        
        # Flatten buffers
        b_obs = obs_buffer.reshape((-1, args.num_agents, obs.shape[-1]))
        b_actions = action_buffer.reshape((-1, args.num_agents, actions[0].shape[-1]))
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = value_buffer.reshape(-1)
        b_log_probs = log_prob_buffer.reshape((-1, args.num_agents))
        
        # Update policy
        for epoch in range(args.update_epochs):
            # Shuffle data
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, b_obs.shape[0])
            batch_size = b_obs.shape[0] // args.num_minibatches
            
            # Update in minibatches
            for start in range(0, b_obs.shape[0], batch_size):
                end = start + batch_size
                mb_inds = permutation[start:end]
                
                # Get minibatch data
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_values = b_values[mb_inds]
                mb_log_probs = b_log_probs[mb_inds]
                
                # Update critic
                def critic_loss_fn(critic_params):
                    # Get global state for each observation
                    mb_global_obs = jnp.zeros((mb_obs.shape[0], env.state(state).shape[-1]))  # Simplified
                    
                    # Compute values
                    values_pred = critic.apply(critic_params, mb_global_obs)
                    
                    # Compute value loss
                    value_pred_clipped = mb_values + jnp.clip(values_pred - mb_values, -args.clip_coef, args.clip_coef)
                    value_losses = jnp.square(values_pred - mb_returns)
                    value_losses_clipped = jnp.square(value_pred_clipped - mb_returns)
                    value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
                    
                    return value_loss
                
                # Update actor
                def actor_loss_fn(actor_params):
                    # Compute actions for all agents
                    loss_actors = 0
                    entropy = 0
                    
                    for agent_id in range(args.num_agents):
                        # Get distribution
                        dist = actor.apply(actor_params, mb_obs[:, agent_id])
                        
                        # Compute log probs and entropy
                        log_prob = dist.log_prob(mb_actions[:, agent_id])
                        entropy += jnp.mean(dist.entropy())
                        
                        # Compute ratio and clipped loss
                        ratio = jnp.exp(log_prob - mb_log_probs[:, agent_id])
                        
                        # Compute surrogate losses
                        loss1 = -mb_advantages * ratio
                        loss2 = -mb_advantages * jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                        loss_actors += jnp.mean(jnp.maximum(loss1, loss2))
                    
                    # Average over agents
                    loss_actors /= args.num_agents
                    entropy /= args.num_agents
                    
                    # Compute total loss
                    value_loss = critic_loss_fn(critic_state.params)
                    total_loss = loss_actors - args.ent_coef * entropy + args.vf_coef * value_loss
                    
                    return total_loss, (loss_actors, entropy, value_loss)
                
                # Compute gradients
                (actor_loss, (policy_loss, entropy, value_loss)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
                critic_grads = jax.grad(critic_loss_fn)(critic_state.params)
                
                # Update parameters
                actor_state = actor_state.apply_gradients(grads=actor_grads)
                critic_state = critic_state.apply_gradients(grads=critic_grads)
        
        # Print progress
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
    
    # Save final model
    os.makedirs(f"results/{args.exp_name}/models", exist_ok=True)
    np.save(
        f"results/{args.exp_name}/models/policy_0.npy",
        jax.device_get(actor_state.params),
    )
    np.save(
        f"results/{args.exp_name}/models/critic_0.npy",
        jax.device_get(critic_state.params),
    )
    np.save(
        f"results/{args.exp_name}/models/weight_0.npy",
        jax.device_get(jnp.array(args.weights)),
    )
    
    print(f"Training complete. Models saved to results/{args.exp_name}/models/")
    print("To visualize policies, run:")
    print(f"  python visualize_catch.py --model-path results/{args.exp_name}/models/policy_0")


if __name__ == "__main__":
    main()