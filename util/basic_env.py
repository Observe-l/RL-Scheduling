import gym
from gym import spaces
import numpy as np
from .multi_discrete import MultiDiscrete
import traci

class MultiAgentEnv(gym.Env):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None):
        
        self.world = world
        self.agents = self.world.agents

        self.n = len(world.agents)
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
            # act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])

            self.action_space.append(tmp_action_space)

            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=0, high=+np.inf, shape=(obs_dim,),dtype=np.float32))
    
    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n':[]}
        # operable agent
        self.agents = self.world.agents
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # 500 seconds SUMO time
        for _ in range(500):
            self.world.step()
        
        self.world.resume_truck()
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))
        
        self.world.park_truck()
        self.world.flag_reset()
        
        return obs_n, reward_n, done_n, info_n


    def reset(self):
        self.reset_callback(self.world)

        obs_n = []
        self.agents = self.world.agents

        self.world.resume_truck()
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        
        self.world.park_truck()

        return obs_n

    def _set_action(self, action, agent):
        factory_agents = self.world.factory_agents()
        if agent.truck:
            target_id = factory_agents[np.argmax(action)].id
            if agent.operable_flag:
                agent.delivery(destination=target_id)
            else:
                pass
        else:
            agent.req_truck = True if action[0] > 0.5 else False

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)
    
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)
    
    # get dones for a particular agent
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)
    

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)
    