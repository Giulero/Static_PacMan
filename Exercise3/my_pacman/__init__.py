from gym.envs.registration import register

register(
    id='my_pacman-v0',
    entry_point='my_pacman.envs:MyPacman',
    # timestep_limit=1000,
    # reward_threshold=1.0,
    nondeterministic = False,
)