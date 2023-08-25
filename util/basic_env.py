import gym
from gym import spaces
import numpy as np
from .multi_discrete import MultiDiscrete

class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None):
        
        self.world = world
        self.agents = self.world.agents
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            if agent.truck:
                # number of factory, the truck's new destination
                tmp_action_space = spaces.Discrete(4)
            else:
                # need truck or not
                tmp_action_space = spaces.Discrete(2)
            total_action_space.append(tmp_action_space)
            act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])

            self.action_space.append(act_space)

            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=0, high=+np.inf, shape=(obs_dim),dtype=np.float32))
    
    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n':[]}
        # operable agent
        self.agents = self.world.agents

    
    def _set_action(self, action, agent, action_space):
        