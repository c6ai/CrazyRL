import numpy as np
import time
from crazy_rl.multi_agent.numpy.catch.catch import Catch

if __name__ == "__main__":
    parallel_env = Catch(
        drone_ids=np.array([0, 1, 2, 3]),
        render_mode=None,  # Set to None to disable rendering
        init_flying_pos=np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [2, 2, 1]]),
        init_target_location=np.array([1, 1, 2.5]),
        target_speed=0.1,
    )

    observations, infos = parallel_env.reset()
    
    # Run for 100 steps
    for _ in range(100):
        if not parallel_env.agents:
            break
            
        actions = {
            agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        print("Step:", _, "rewards:", rewards)