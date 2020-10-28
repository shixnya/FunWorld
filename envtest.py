import gym

env = gym.make('gym_smallworld:smallworld-v0')


env.reset()
env.render()

for i in range(50):
    env.step(1)
    env.render()

env.reset()
env.render()

env.close()