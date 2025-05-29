import numpy as np
import time
from crazy_rl.multi_agent.numpy.circle.circle import Circle

if __name__ == "__main__":
    parallel_env = Circle(
        drone_ids=np.array([0, 1, 2]),
        render_mode=None,  # Set to None to disable rendering
        init_flying_pos=np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1]]),
    )

    observations, infos = parallel_env.reset()
    
    # Run for 100 steps
    for step in range(100):
        if not parallel_env.agents:
            break
            
        actions = {
            agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        
        if step % 10 == 0:  # Print every 10 steps to reduce output
            print(f"Step: {step}, rewards: {rewards}")