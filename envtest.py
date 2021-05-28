import gym
import time
from tqdm import trange
import numpy as np

watch = False  # if True, it slows down playback


print("Initializing the environment")
env = gym.make("gym_smallworld:smallworld-v0")
env.reset()
env.render()
action_num = env.action_space.n


render_options = [False, True]

print("randomly explore the environment with and without rendering, check mean rewards")
for i in range(len(render_options)):
    render = render_options[i]
    pos_count = 0  # positive reward count
    total_reward = 0
    for j in trange(2002, desc=f"trial {i}, rendering: {render}"):
        action = np.random.randint(action_num)
        obs, reward, done, _ = env.step(action)
        # time.sleep(0.001)  # needed?
        total_reward += reward
        if reward > 0:
            pos_count += 1

        if render:
            env.render()
            if watch:
                time.sleep(0.01)

        if done:
            # show information
            env.reset()
            break
    print(f"*** # of positive rewards: {pos_count}")
    print(f"*** total rewards: {total_reward}")


env.close()

# let's implement a better testing
# random agent
