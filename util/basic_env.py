import gym
from gym import spaces
import numpy as np

class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        
        self.world = world
        self.agents = self.world