from gym.envs.registration import register

register(
    id='smallworld-v0',
    entry_point='gym_smallworld.envs:SmallWorldEnv',
)

#register(
#    id='smallworld-hard-v0',
#    entry_point='gym_smallworld.envs:SmallWorldHardEnv',
#)