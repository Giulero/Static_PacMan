import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# '#' are walls
# 'O' are ghosts
# '.' stuff to eat
# '_' empty space

MAP = [
    "##########",
    "#..O.....#",
    "#.###.##.#",
    "#.#....#.#",
    "#.#.O..#.#",
    "#.#....#.#",
    "#.#....#.#",
    "#.##.###.#",
    "#......O.#",
    "##########",
]

MAP_STATIC = [
    "##########",
    "#__O_____#",
    "#_###_##_#",
    "#_#____#_#",
    "#_#_O__#_#",
    "#_#____#_#",
    "#_#____#_#",
    "#_##_###_#",
    "#G_____O_#",
    "##########",
]

class MyPacman(gym.Env):
  """
  Pacman env
  10x10 map
  Possible actions:
  - 0: rest
  - 1: move left
  - 2: move right
  - 3: move up
  - 4: move down
  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.map = np.asarray(MAP,dtype='c')
    self.pos = [5,5]
    self.map[5,5] = 'P'
    self.reward = 0
    self.lives = 3
    self.done = False

  def reset(self):
    # self.map = np.asarray(MAP,dtype='c')
    # if you are here it's because you have found a ghost :(
    self.map[self.pos[0]][self.pos[1]] = 'O'
    self.pos = [5,5]
    self.map[5,5] = 'P'
    self.done = False

  def randomAction(self):
    return random.sample([0,1,2,3,4],1)
    
  def update(self, action):
    action = action[0]
    prev_pos = np.copy(self.pos)
    if action == 1:
      if self.map[self.pos[0]][self.pos[1]-1] not in ['#']:
        self.pos[1] -= 1
    if action == 2:
      if self.map[self.pos[0]][self.pos[1]+1] not in ['#']:
        self.pos[1] += 1
    if action == 3:
      if self.map[self.pos[0]-1][self.pos[1]] not in ['#']:
        self.pos[0] -= 1
    if action == 4:
      if self.map[self.pos[0]+1][self.pos[1]] not in ['#']:
        self.pos[0] += 1
    if self.map[self.pos[0]][self.pos[1]] in ['O']:
      self.lives -= 1
      self.reward -= 100
      self.done = True
    elif self.map[self.pos[0]][self.pos[1]] in ['#']:
      self.pos = prev_pos
    elif self.map[self.pos[0]][self.pos[1]] in ['.']:
      self.reward += 1
      self.map[prev_pos[0]][prev_pos[1]] = '_'
      self.map[self.pos[0]][self.pos[1]] = 'P'
    elif self.map[self.pos[0]][self.pos[1]] in ['G']:
      self.reward += 100
      self.map[prev_pos[0]][prev_pos[1]] = '_'
      self.map[self.pos[0]][self.pos[1]] = 'P'
      self.done = True
    else:
      self.reward -= 1
      self.map[prev_pos[0]][prev_pos[1]] = '_'
      self.map[self.pos[0]][self.pos[1]] = 'P'

  def transition(self, action, pos_):
    pos = np.copy(pos_)
    prev_pos = np.copy(pos_)
    action = action[0]
    if action == 1:
      pos[1] -= 1
    if action == 2:
      pos[1] += 1
    if action == 3:
      pos[0] -= 1
    if action == 4:
      pos[0] += 1
    if self.map[pos[0],pos[1]] in ['#']:
      pos = prev_pos
    return pos

        
  def getGhosts(self):
    ghosts_pos = []
    for i in range(10):
      for j in range(10):
        if self.map[i][j] in ['O']:
          ghosts_pos.append([i, j])
    return ghosts_pos

  def step(self, action):
    ''' must return 
    - observation
    - reward
    - done
    - info'''
    self.update(action)
    ghosts = self.getGhosts()
    observation = {'ghosts_pos': ghosts, 'num_lives': self.lives, 'pac_pos': self.pos}
    reward = self.reward
    done = self.done
    print('Lives {} || Reward {}'.format(self.lives, self.reward))
    return observation, reward, done, observation['num_lives']

  def render(self):
    print('PacMan World')
    print(self.map)

  def getUtilityOnMatrix(self, utility_matrix, pos):
    return utility_matrix[pos[0]][pos[1]]

  def getStates(self):
    states = []
    for i in range(10):
      for j in range(10):
        if self.map[i][j] not in ['#']:
          states.append([i, j])
    return states

  def getGoalStates(self):
    states = []
    for i in range(10):
      for j in range(10):
        if self.map[i][j] in ['G', 'O']:
          states.append([i, j])
    return states

  def getRewardMatrix(self):
    reward_matrix = np.zeros(self.map.shape)
    for i in range(10):
      for j in range(10):
        if self.map[i,j] in ['G']:
          reward_matrix[i,j] = 100
        if self.map[i,j] in ['O']:
          reward_matrix[i,j] = -100
    return reward_matrix

  def valueIteration(self):
    self.map = np.asarray(MAP_STATIC,dtype='c')
    gamma = 0.85
    delta = 100.0
    epsilon = 1e-20
    states = self.getStates()

    goalStates = self.getGoalStates()

    utility_matrix_o = np.zeros(self.map.shape)
    utility_matrix_n = np.zeros(self.map.shape)

    reward_matrix = self.getRewardMatrix()
    t = 0
    while delta > epsilon*(1 - gamma)/gamma:
      delta = 0.0
      utility_matrix_o = np.copy(utility_matrix_n)
      for s in states:
        util_vec = [self.getUtilityOnMatrix(utility_matrix_o, self.transition([1], s)),
                    self.getUtilityOnMatrix(utility_matrix_o, self.transition([2], s)),
                    self.getUtilityOnMatrix(utility_matrix_o, self.transition([3], s)),
                    self.getUtilityOnMatrix(utility_matrix_o, self.transition([4], s))]
        utility_matrix_n[s[0],s[1]] = reward_matrix[s[0],s[1]] +  gamma*max(util_vec)
        if s in goalStates:
          utility_matrix_n[s[0],s[1]] = reward_matrix[s[0],s[1]]
        delta = max(delta, abs(utility_matrix_n[s[0],s[1]] - utility_matrix_o[s[0],s[1]]))
        t+=1
      if delta < epsilon*(1 - gamma)/gamma:
        print('Converged after {} iterations'.format(t))

    print(utility_matrix_n.round(1))
    fig, ax = plt.subplots()
    im = ax.imshow(utility_matrix_n)
    plt.title('Value iteration MAP')
    plt.show()
    