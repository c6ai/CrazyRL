"""
Self-contained implementation of the Catch environment for multi-objective reinforcement learning.
This version includes realistic physics and smooth trajectories for intruders and evader craft.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any

# Constants
CLOSENESS_THRESHOLD = 0.2
DEFAULT_SEED = 42

class CatchEnv:
    """
    A self-contained implementation of the Catch environment where multiple agents
    learn to surround a moving target (evader) while maintaining distance from each other.
    
    This environment supports multi-objective reinforcement learning with two objectives:
    1. Minimize distance to target (get close to target)
    2. Maximize distance from other agents (stay far from others)
    """
    
    def __init__(
        self,
        num_agents: int = 4,
        size: float = 3.0,
        target_speed: float = 0.1,
        multi_obj: bool = True,
        max_steps: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        physics_enabled: bool = True,
        smooth_trajectories: bool = True,
    ):
        """
        Initialize the Catch environment.
        
        Args:
            num_agents: Number of agents (drones)
            size: Size of the environment (defines boundaries)
            target_speed: Speed of the target (evader)
            multi_obj: Whether to return multi-objective rewards
            max_steps: Maximum number of steps before truncation
            seed: Random seed for reproducibility
            render_mode: Rendering mode (None for headless)
            physics_enabled: Whether to use realistic physics
            smooth_trajectories: Whether to use smooth trajectories
        """
        self.num_agents = num_agents
        self.size = size
        self.target_speed = target_speed
        self.multi_obj = multi_obj
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.physics_enabled = physics_enabled
        self.smooth_trajectories = smooth_trajectories
        
        # Set random seed
        self.seed(seed if seed is not None else DEFAULT_SEED)
        
        # Initialize agent names
        self.agent_names = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agent_names.copy()
        self.agents = []
        
        # Initialize positions and velocities
        self.agent_positions = {}
        self.agent_velocities = {}
        self.previous_positions = {}
        self.target_position = np.zeros(3)
        self.target_velocity = np.zeros(3)
        self.previous_target = np.zeros(3)
        
        # Initialize default positions
        self._init_default_positions()
        
        # Initialize step counter
        self.timestep = 0
        
        # Physics parameters
        if self.physics_enabled:
            self.mass = 0.1  # kg
            self.drag_coefficient = 0.3
            self.max_force = 1.0  # N
            self.dt = 0.1  # seconds
        
        # Observation and action spaces
        self.observation_dim = 3 * (num_agents + 1)  # positions of all agents and target
        self.action_dim = 3  # 3D movement
        
        # Reward parameters
        self.collision_penalty = -10.0
        self.out_of_bounds_penalty = -10.0
        self.catch_reward = 10.0
        
        # Termination flags
        self.terminated = {agent: False for agent in self.agent_names}
        self.truncated = {agent: False for agent in self.agent_names}
    
    def _init_default_positions(self):
        """Initialize default positions for agents and target."""
        # Initialize target at center, slightly elevated
        self.target_position = np.array([0.0, 0.0, 1.5])
        
        # Initialize agents in a circle around the target
        radius = 2.0
        for i, agent in enumerate(self.agent_names):
            angle = 2 * np.pi * i / self.num_agents
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 1.0
            self.agent_positions[agent] = np.array([x, y, z])
            self.agent_velocities[agent] = np.zeros(3)
    
    def seed(self, seed: int = None):
        """Set random seed for reproducibility."""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observations: Dictionary of observations for each agent
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset step counter
        self.timestep = 0
        
        # Reset agents list
        self.agents = self.possible_agents.copy()
        
        # Reset positions with some randomness
        self._init_default_positions()
        
        # Add some random perturbation to initial positions
        for agent in self.agent_names:
            perturbation = self.np_random.uniform(-0.5, 0.5, 3)
            perturbation[2] = self.np_random.uniform(0, 0.3)  # Less vertical perturbation
            self.agent_positions[agent] += perturbation
            
            # Ensure within bounds
            self.agent_positions[agent] = np.clip(
                self.agent_positions[agent],
                [-self.size, -self.size, 0.2],
                [self.size, self.size, self.size]
            )
            
            # Reset velocities
            self.agent_velocities[agent] = np.zeros(3)
        
        # Add some random perturbation to target position
        perturbation = self.np_random.uniform(-0.5, 0.5, 3)
        perturbation[2] = self.np_random.uniform(0, 0.3)  # Less vertical perturbation
        self.target_position += perturbation
        self.target_position = np.clip(
            self.target_position,
            [-self.size, -self.size, 0.5],
            [self.size, self.size, self.size]
        )
        
        # Reset target velocity
        self.target_velocity = np.zeros(3)
        
        # Store previous positions for reward calculation
        self.previous_positions = {agent: self.agent_positions[agent].copy() for agent in self.agent_names}
        self.previous_target = self.target_position.copy()
        
        # Reset termination flags
        self.terminated = {agent: False for agent in self.agent_names}
        self.truncated = {agent: False for agent in self.agent_names}
        
        # Compute observations
        observations = self._compute_observations()
        
        return observations, {}
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            terminated: Dictionary of termination flags for each agent
            truncated: Dictionary of truncation flags for each agent
            info: Additional information
        """
        # Increment step counter
        self.timestep += 1
        
        # Store previous positions for reward calculation
        self.previous_positions = {agent: self.agent_positions[agent].copy() for agent in self.agent_names}
        self.previous_target = self.target_position.copy()
        
        # Move target
        self._move_target()
        
        # Move agents based on actions
        self._move_agents(actions)
        
        # Check for terminations
        self._check_terminations()
        
        # Check for truncations
        self._check_truncations()
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Compute observations
        observations = self._compute_observations()
        
        # Update agents list (remove terminated agents)
        self.agents = [agent for agent in self.agents if not self.terminated[agent] and not self.truncated[agent]]
        
        return observations, rewards, self.terminated, self.truncated, {}
    
    def _move_target(self):
        """Move the target (evader) based on agent positions."""
        if not self.smooth_trajectories:
            # Simple target movement (original implementation)
            # Calculate mean position of agents
            mean_pos = np.zeros(3)
            for agent in self.agents:
                mean_pos += self.agent_positions[agent]
            
            if self.agents:
                mean_pos /= len(self.agents)
            
            # Calculate distance to mean position
            dist = np.linalg.norm(mean_pos - self.target_position)
            
            # Move away from agents if they're close
            if dist > 0.2:
                direction = (self.target_position - mean_pos) / dist
                self.target_position += direction * self.target_speed
            else:
                # Random movement if agents are too close
                self.target_position += self.np_random.uniform(-1, 1, 3) * self.target_speed * 0.1
        else:
            # Advanced target movement with smooth trajectories
            # Calculate mean position of agents
            mean_pos = np.zeros(3)
            for agent in self.agents:
                mean_pos += self.agent_positions[agent]
            
            if self.agents:
                mean_pos /= len(self.agents)
            
            # Calculate desired direction (away from agents)
            dist = np.linalg.norm(mean_pos - self.target_position)
            
            if dist > 0.2:
                # Move away from agents
                desired_direction = (self.target_position - mean_pos) / dist
            else:
                # Random direction if agents are too close
                desired_direction = self.np_random.uniform(-1, 1, 3)
                desired_direction = desired_direction / np.linalg.norm(desired_direction)
            
            # Add some randomness to make movement less predictable
            noise = self.np_random.uniform(-0.3, 0.3, 3)
            desired_direction += noise
            desired_direction = desired_direction / np.linalg.norm(desired_direction)
            
            # Calculate desired velocity
            desired_velocity = desired_direction * self.target_speed
            
            # Smooth velocity transition (acceleration-based)
            acceleration = (desired_velocity - self.target_velocity) * 2.0  # Adjust factor for responsiveness
            self.target_velocity += acceleration * 0.1  # dt
            
            # Limit maximum velocity
            speed = np.linalg.norm(self.target_velocity)
            if speed > self.target_speed * 1.5:
                self.target_velocity = self.target_velocity / speed * self.target_speed * 1.5
            
            # Update position
            self.target_position += self.target_velocity
        
        # Ensure target stays within bounds
        self.target_position = np.clip(
            self.target_position,
            [-self.size, -self.size, 0.2],
            [self.size, self.size, self.size]
        )
    
    def _move_agents(self, actions: Dict[str, np.ndarray]):
        """
        Move agents based on actions.
        
        Args:
            actions: Dictionary of actions for each agent
        """
        for agent in self.agents:
            if agent in actions:
                action = np.clip(actions[agent], -1.0, 1.0)
                
                if not self.physics_enabled:
                    # Simple movement (original implementation)
                    # Scale action to reasonable movement
                    movement = action * 0.2
                    
                    # Update position
                    self.agent_positions[agent] += movement
                else:
                    # Physics-based movement
                    # Convert action to force
                    force = action * self.max_force
                    
                    # Calculate acceleration (F = ma)
                    acceleration = force / self.mass
                    
                    # Apply drag (proportional to velocity squared)
                    velocity_magnitude = np.linalg.norm(self.agent_velocities[agent])
                    if velocity_magnitude > 0:
                        drag_direction = -self.agent_velocities[agent] / velocity_magnitude
                        drag_magnitude = self.drag_coefficient * velocity_magnitude**2
                        drag_acceleration = drag_direction * drag_magnitude / self.mass
                        acceleration += drag_acceleration
                    
                    # Update velocity (v = v0 + a*t)
                    self.agent_velocities[agent] += acceleration * self.dt
                    
                    # Limit maximum velocity
                    velocity_magnitude = np.linalg.norm(self.agent_velocities[agent])
                    max_velocity = 1.0
                    if velocity_magnitude > max_velocity:
                        self.agent_velocities[agent] = self.agent_velocities[agent] / velocity_magnitude * max_velocity
                    
                    # Update position (x = x0 + v*t)
                    self.agent_positions[agent] += self.agent_velocities[agent] * self.dt
                
                # Ensure agent stays within bounds
                self.agent_positions[agent] = np.clip(
                    self.agent_positions[agent],
                    [-self.size, -self.size, 0.0],
                    [self.size, self.size, self.size]
                )
    
    def _check_terminations(self):
        """Check for termination conditions."""
        for agent in self.agents:
            # Check for collisions with other agents
            for other_agent in self.agents:
                if other_agent != agent:
                    distance = np.linalg.norm(self.agent_positions[agent] - self.agent_positions[other_agent])
                    if distance < CLOSENESS_THRESHOLD:
                        self.terminated[agent] = True
            
            # Check for collision with ground
            if self.agent_positions[agent][2] < CLOSENESS_THRESHOLD:
                self.terminated[agent] = True
            
            # Check for collision with target
            distance_to_target = np.linalg.norm(self.agent_positions[agent] - self.target_position)
            if distance_to_target < CLOSENESS_THRESHOLD:
                self.terminated[agent] = True
    
    def _check_truncations(self):
        """Check for truncation conditions."""
        if self.timestep >= self.max_steps:
            for agent in self.agents:
                self.truncated[agent] = True
    
    def _compute_rewards(self) -> Dict[str, np.ndarray]:
        """
        Compute rewards for each agent.
        
        Returns:
            Dictionary of rewards for each agent
        """
        rewards = {}
        
        for agent in self.agent_names:
            # Skip if agent is already terminated
            if agent not in self.agents:
                if self.multi_obj:
                    rewards[agent] = np.array([0.0, 0.0])
                else:
                    rewards[agent] = 0.0
                continue
            
            # Calculate reward for being close to target
            dist_to_target = np.linalg.norm(self.agent_positions[agent] - self.target_position)
            prev_dist_to_target = np.linalg.norm(self.previous_positions[agent] - self.previous_target)
            reward_close_to_target = prev_dist_to_target - dist_to_target  # Positive if getting closer
            
            # Calculate reward for being far from other agents
            reward_far_from_others = 0.0
            count = 0
            for other_agent in self.agent_names:
                if other_agent != agent and other_agent in self.agents:
                    distance = np.linalg.norm(self.agent_positions[agent] - self.agent_positions[other_agent])
                    reward_far_from_others += distance
                    count += 1
            
            if count > 0:
                reward_far_from_others /= count
            
            # Check for collisions with other agents
            for other_agent in self.agent_names:
                if other_agent != agent and other_agent in self.agents:
                    distance = np.linalg.norm(self.agent_positions[agent] - self.agent_positions[other_agent])
                    if distance < CLOSENESS_THRESHOLD:
                        reward_close_to_target = self.collision_penalty
                        reward_far_from_others = self.collision_penalty
            
            # Check for collision with ground
            if self.agent_positions[agent][2] < CLOSENESS_THRESHOLD:
                reward_close_to_target = self.out_of_bounds_penalty
                reward_far_from_others = self.out_of_bounds_penalty
            
            # Check for collision with target
            distance_to_target = np.linalg.norm(self.agent_positions[agent] - self.target_position)
            if distance_to_target < CLOSENESS_THRESHOLD:
                reward_close_to_target = self.catch_reward
                reward_far_from_others = self.collision_penalty
            
            # Return multi-objective or scalarized reward
            if self.multi_obj:
                rewards[agent] = np.array([reward_close_to_target, reward_far_from_others])
            else:
                # Scalarized reward with default weights
                rewards[agent] = 0.9995 * reward_close_to_target + 0.0005 * reward_far_from_others
        
        return rewards
    
    def _compute_observations(self) -> Dict[str, np.ndarray]:
        """
        Compute observations for each agent.
        
        Returns:
            Dictionary of observations for each agent
        """
        observations = {}
        
        for agent in self.agent_names:
            # Agent's own position
            obs = self.agent_positions[agent].copy()
            
            # Target position
            obs = np.append(obs, self.target_position)
            
            # Other agents' positions
            for other_agent in self.agent_names:
                if other_agent != agent:
                    obs = np.append(obs, self.agent_positions[other_agent])
            
            observations[agent] = obs
        
        return observations
    
    def get_agent_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all agents."""
        return {agent: pos.copy() for agent, pos in self.agent_positions.items()}
    
    def get_target_position(self) -> np.ndarray:
        """Get current position of the target."""
        return self.target_position.copy()
    
    def get_agent_velocities(self) -> Dict[str, np.ndarray]:
        """Get current velocities of all agents."""
        return {agent: vel.copy() for agent, vel in self.agent_velocities.items()}
    
    def get_target_velocity(self) -> np.ndarray:
        """Get current velocity of the target."""
        return self.target_velocity.copy()
    
    def close(self):
        """Clean up resources."""
        pass


class MOPolicy:
    """
    A simple multi-objective policy for the Catch environment.
    This policy balances between getting close to the target and staying away from other agents.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        weights: List[float] = [0.8, 0.2],
        seed: Optional[int] = None
    ):
        """
        Initialize the policy.
        
        Args:
            observation_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            weights: Weights for the objectives [close_to_target, far_from_others]
            seed: Random seed for reproducibility
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.weights = np.array(weights) / sum(weights)  # Normalize weights
        self.np_random = np.random.RandomState(seed if seed is not None else DEFAULT_SEED)
        
        # Initialize simple linear policy parameters
        self.weights_close = self.np_random.uniform(-0.1, 0.1, (action_dim, observation_dim))
        self.weights_far = self.np_random.uniform(-0.1, 0.1, (action_dim, observation_dim))
        self.bias_close = self.np_random.uniform(-0.1, 0.1, action_dim)
        self.bias_far = self.np_random.uniform(-0.1, 0.1, action_dim)
        
        # Set specific weights for target-seeking behavior
        # Agent position to target vector
        for i in range(action_dim):
            # Position of agent (first 3 elements)
            self.weights_close[i, i] = -1.0
            # Position of target (next 3 elements)
            self.weights_close[i, i + 3] = 1.0
        
        # Set specific weights for agent-avoiding behavior
        for i in range(action_dim):
            # For each other agent
            for j in range(1, 4):  # Assuming 4 agents total
                # Position of other agents
                other_agent_idx = 3 + 3 + (j - 1) * 3
                if other_agent_idx + i < observation_dim:
                    self.weights_far[i, other_agent_idx + i] = -1.0
    
    def act(self, observation: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Generate an action based on the observation.
        
        Args:
            observation: The current observation
            noise_scale: Scale of exploration noise
            
        Returns:
            action: The action to take
        """
        # Compute actions for each objective
        action_close = np.dot(self.weights_close, observation) + self.bias_close
        action_far = np.dot(self.weights_far, observation) + self.bias_far
        
        # Combine actions based on weights
        action = self.weights[0] * action_close + self.weights[1] * action_far
        
        # Add exploration noise
        noise = self.np_random.normal(0, noise_scale, self.action_dim)
        action += noise
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action


def create_diverse_policies(
    num_policies: int,
    observation_dim: int,
    action_dim: int,
    seed: Optional[int] = None
) -> List[MOPolicy]:
    """
    Create a diverse set of policies with different objective weights.
    
    Args:
        num_policies: Number of policies to create
        observation_dim: Dimension of the observation space
        action_dim: Dimension of the action space
        seed: Random seed for reproducibility
        
    Returns:
        List of policies
    """
    np_random = np.random.RandomState(seed if seed is not None else DEFAULT_SEED)
    policies = []
    
    # Create policies with different weights
    for i in range(num_policies):
        # Generate weights that sum to 1
        if i == 0:
            # First policy prioritizes getting close to target
            weights = [0.95, 0.05]
        elif i == num_policies - 1:
            # Last policy prioritizes staying away from others
            weights = [0.05, 0.95]
        else:
            # Intermediate policies with varying weights
            alpha = i / (num_policies - 1)
            weights = [1.0 - alpha, alpha]
        
        # Create policy with these weights
        policy = MOPolicy(
            observation_dim=observation_dim,
            action_dim=action_dim,
            weights=weights,
            seed=np_random.randint(0, 10000)
        )
        policies.append(policy)
    
    return policies