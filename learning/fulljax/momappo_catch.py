"""Multi-Objective MAPPO implementation for the Catch environment."""
import os
import time
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from crazy_rl.utils.jax_wrappers import (
    AddIDToObs,
    AutoReset,
    ClipActions,
    LinearizeReward,
    LogWrapper,
    NormalizeObservation,
    NormalizeVecReward,
    VecEnv,
)
from learning.fulljax.momappo_fulljax import Actor, Critic, equally_spaced_weights


def make_train(args):
    """Create a training function for multi-objective MAPPO."""
    num_updates = args.total_timesteps // args.num_steps // args.num_envs
    minibatch_size = args.num_envs * args.num_steps // args.num_minibatches

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
        return args.lr * frac

    def train(key: chex.PRNGKey, weights: jnp.ndarray):
        # Create environment using the provided env_fn
        env = args.env_fn()
        
        # Apply wrappers
        env = ClipActions(env)
        env = NormalizeObservation(env)
        env = AddIDToObs(env, args.num_agents)
        env = LogWrapper(env, reward_dim=3)  # Catch has 3 objectives
        env = LinearizeReward(env, weights)
        env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
        env = VecEnv(env)  # vmaps the env public methods
        env = NormalizeVecReward(env, args.gamma)

        # Initial reset to have correct dimensions in the observations
        obs, info, state = env.reset(key=jax.random.PRNGKey(args.seed))

        # INIT NETWORKS
        actor = Actor(env.action_space(0).shape[0], activation=args.activation)
        critic = Critic(activation=args.activation)
        key, actor_key, critic_key = jax.random.split(key, 3)
        dummy_local_obs_and_id = jnp.zeros(env.observation_space(0).shape[0] + args.num_agents)
        dummy_global_obs = jnp.zeros(env.state(state).shape)
        actor_params = actor.init(actor_key, dummy_local_obs_and_id)
        critic_params = critic.init(critic_key, dummy_global_obs)
        
        if args.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5),
            )

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=tx,
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=tx,
        )

        # INIT BUFFER
        class Trajectory(NamedTuple):
            obs: jnp.ndarray
            action: jnp.ndarray
            reward: jnp.ndarray
            termination: jnp.ndarray
            truncation: jnp.ndarray
            value: jnp.ndarray
            log_prob: jnp.ndarray
            gae: jnp.ndarray
            global_obs: jnp.ndarray

        class RunnerState(NamedTuple):
            actor_state: TrainState
            critic_state: TrainState
            env_state: Any
            last_obs: jnp.ndarray
            key: chex.PRNGKey
            global_obs: jnp.ndarray

        runner_state = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            env_state=state,
            last_obs=obs,
            key=key,
            global_obs=env.state(state),
        )

        def _env_step(runner_state, _):
            actor_state, critic_state, env_state, last_obs, key, global_obs = runner_state

            # SELECT ACTION
            key, subkey = jax.random.split(key)
            keys_step = jax.random.split(subkey, args.num_envs)
            
            # Add agent ID to observation
            agent_ids = jnp.arange(args.num_agents)
            agent_ids_one_hot = jnp.eye(args.num_agents)
            
            # Compute actions for all agents
            def _get_action_and_log_prob(agent_id, obs):
                obs_with_id = jnp.concatenate([obs, agent_ids_one_hot[agent_id]], axis=-1)
                dist = actor.apply(actor_state.params, obs_with_id)
                action = dist.sample(seed=subkey)
                log_prob = dist.log_prob(action)
                return action, log_prob
            
            actions = []
            log_probs = []
            
            for agent_id in range(args.num_agents):
                action, log_prob = _get_action_and_log_prob(agent_id, last_obs[:, agent_id])
                actions.append(action)
                log_probs.append(log_prob)
            
            joint_actions = jnp.stack(actions, axis=1)
            log_probs = jnp.stack(log_probs, axis=1)
            
            # Compute values
            values = critic.apply(critic_state.params, global_obs)

            # STEP ENV
            obs, rewards, terminateds, truncateds, info, env_states = env.step(env_state, joint_actions, jnp.stack(keys_step))
            global_obs = env.state(env_states)
            
            traj_batch = Trajectory(
                obs=last_obs,
                action=joint_actions,
                reward=rewards,
                termination=terminateds,
                truncation=truncateds,
                value=values,
                log_prob=log_probs,
                gae=jnp.zeros_like(rewards),
                global_obs=global_obs,
            )

            runner_state = RunnerState(
                actor_state=actor_state,
                critic_state=critic_state,
                env_state=env_states,
                last_obs=obs,
                key=key,
                global_obs=global_obs,
            )

            return runner_state, traj_batch

        def _compute_gae(traj_batch, last_val):
            def _get_gae(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                value, reward, termination, truncation = transition
                done = termination | truncation

                delta = reward + args.gamma * next_value * (1 - done) - value
                gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, gae = jax.lax.scan(
                _get_gae,
                (jnp.zeros_like(traj_batch.reward), last_val),
                (traj_batch.value, traj_batch.reward, traj_batch.termination, traj_batch.truncation),
                reverse=True,
                unroll=16,
            )
            return gae

        def _update_epoch(update_state, _):
            def _update_minbatch(train_states, batch_info):
                actor_state, critic_state = train_states
                traj_batch, idxes = batch_info

                # Compute values and returns
                values = critic.apply(critic_state.params, traj_batch.global_obs)
                returns = traj_batch.gae + traj_batch.value

                # Update critic
                def _critic_loss_fn(critic_params):
                    values_pred = critic.apply(critic_params, traj_batch.global_obs)
                    value_pred_clipped = traj_batch.value + (values_pred - traj_batch.value).clip(-args.clip_eps, args.clip_eps)
                    value_losses = jnp.square(values_pred - returns)
                    value_losses_clipped = jnp.square(value_pred_clipped - returns)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    return value_loss, (value_loss,)

                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                (critic_loss, (value_loss)), critic_grads = critic_grad_fn(critic_state.params)
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                # Update actor
                def _actor_loss_fn(actor_params):
                    # Compute actions for all agents
                    agent_ids = jnp.arange(args.num_agents)
                    agent_ids_one_hot = jnp.eye(args.num_agents)
                    
                    loss_actors = 0
                    entropy = 0
                    
                    for agent_id in range(args.num_agents):
                        obs_with_id = jnp.concatenate([traj_batch.obs[:, agent_id], agent_ids_one_hot[agent_id]], axis=-1)
                        dist = actor.apply(actor_params, obs_with_id)
                        log_prob = dist.log_prob(traj_batch.action[:, agent_id])
                        entropy += dist.entropy().mean()
                        
                        ratio = jnp.exp(log_prob - traj_batch.log_prob[:, agent_id])
                        gae = traj_batch.gae
                        
                        loss_actor1 = -ratio * gae
                        loss_actor2 = -jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * gae
                        loss_actors += jnp.maximum(loss_actor1, loss_actor2).mean()
                    
                    # Average over agents
                    loss_actors /= args.num_agents
                    entropy /= args.num_agents
                    
                    total_loss = loss_actors + args.vf_coef * value_loss - args.ent_coef * entropy
                    return total_loss, (loss_actors, entropy)

                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                (actor_loss, (loss_actors, entropy)), actor_grads = actor_grad_fn(actor_state.params)
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                metrics = {
                    "total_loss": actor_loss + critic_loss,
                    "actor_loss": loss_actors,
                    "critic_loss": value_loss,
                    "entropy": entropy,
                }

                return (actor_state, critic_state), metrics

            runner_state, traj_batch, key = update_state
            
            # Compute GAE
            last_val = critic.apply(runner_state.critic_state.params, runner_state.global_obs)
            traj_batch = traj_batch._replace(gae=_compute_gae(traj_batch, last_val))
            
            # Flatten batch
            batch_size = minibatch_size * args.num_minibatches
            assert batch_size == args.num_steps * args.num_envs, "batch size must be equal to number of steps * number of envs"
            
            # Prepare indices for minibatches
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, batch_size)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
            )
            
            # Update in minibatches
            def _scan_minibatch(carry, idx_start):
                train_states, metrics = carry
                idxes = permutation[idx_start:idx_start + minibatch_size]
                minibatch = jax.tree_util.tree_map(lambda x: jnp.take(x, idxes, axis=0), batch)
                train_states, metrics_minibatch = _update_minbatch(train_states, (minibatch, idxes))
                metrics = jax.tree_util.tree_map(lambda x, y: x + y, metrics, metrics_minibatch)
                return (train_states, metrics), None
            
            # Initialize metrics
            metrics_init = {
                "total_loss": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
            }
            
            # Scan over minibatches
            (train_states, metrics), _ = jax.lax.scan(
                _scan_minibatch,
                ((runner_state.actor_state, runner_state.critic_state), metrics_init),
                jnp.arange(0, batch_size, minibatch_size),
            )
            
            # Average metrics over minibatches
            metrics = jax.tree_util.tree_map(lambda x: x / args.num_minibatches, metrics)
            
            # Update runner state
            runner_state = runner_state._replace(
                actor_state=train_states[0],
                critic_state=train_states[1],
                key=key,
            )
            
            return (runner_state, traj_batch, key), metrics

        def _update_step(runner_state, _):
            # Collect trajectories
            key, subkey = jax.random.split(runner_state.key)
            runner_state = runner_state._replace(key=subkey)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, args.num_steps)
            
            # Update policy and value networks
            update_state = (runner_state, traj_batch, key)
            update_state, metrics = jax.lax.scan(_update_epoch, update_state, None, args.update_epochs)
            runner_state = update_state[0]
            
            # Save model
            def _save_model(runner_state, weight_idx):
                os.makedirs(f"results/{args.exp_name}/models", exist_ok=True)
                np.save(
                    f"results/{args.exp_name}/models/policy_{weight_idx}.npy",
                    jax.device_get(runner_state.actor_state.params),
                )
                np.save(
                    f"results/{args.exp_name}/models/critic_{weight_idx}.npy",
                    jax.device_get(runner_state.critic_state.params),
                )
                np.save(
                    f"results/{args.exp_name}/models/weight_{weight_idx}.npy",
                    jax.device_get(weights),
                )
            
            # Find the index of the current weight vector
            weight_idx = jnp.where(jnp.all(weights == weights[0]))[0][0]
            jax.debug.callback(_save_model, runner_state, weight_idx)
            
            return runner_state, metrics

        # Run training
        print(f"Starting training with weight vector: {weights}")
        start_time = time.time()
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Save final model
        os.makedirs(f"results/{args.exp_name}/models", exist_ok=True)
        weight_idx = jnp.where(jnp.all(weights == weights[0]))[0][0]
        np.save(
            f"results/{args.exp_name}/models/policy_{weight_idx}.npy",
            jax.device_get(runner_state.actor_state.params),
        )
        np.save(
            f"results/{args.exp_name}/models/critic_{weight_idx}.npy",
            jax.device_get(runner_state.critic_state.params),
        )
        np.save(
            f"results/{args.exp_name}/models/weight_{weight_idx}.npy",
            jax.device_get(weights),
        )
        
        return metrics

    return train