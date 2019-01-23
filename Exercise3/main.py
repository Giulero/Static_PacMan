import gym
import my_pacman
import numpy as np
env = gym.make('my_pacman-v0')

print('random actions')
for t in range(10):
    action = env.randomAction()
    # print("action {}".format(action))
    observation, reward, done, num_lives = env.step(action)
    env.render()
    print('OBSERVATION', observation)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        env.reset()

print('value iteration')
env.valueIteration()