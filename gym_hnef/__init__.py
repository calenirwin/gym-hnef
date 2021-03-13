from gym.envs.registration import register

register(
    id='hnef-v0',
    entry_point='gym_hnef.envs:HnefEnv',
)