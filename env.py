import gym
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from csv import writer
from pathlib import Path
import numpy as np
import traci
import sys
import optparse
import random

from util.lorry_shcedule import Lorry
from util.factory import Factory
from util.product import product_management

class sumoEnv(MultiAgentEnv):
    '''
    Ray Multiagent environment
    Designed for Truck schduling RL algorithm
    Simple truck model without fault, simple factory model
    '''
    def __init__(self, env_config:dict):
        # 12 lorries
        self.config = env_config
        self.lorry_num =  12
        self.path = f'/home/lwh/Documents/Code/PredM/result/' + self.config['algo']

        self.num_cpu = "24"
        self.map_file = "map/3km_1week/osm.sumocfg"

        # Create folder
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.lorry_file = self.path + '/lorry_record.csv'
        self.result_file = self.path + '/result.csv'
        self.reward_file = self.path + '/reward.csv'

        '''
        Top level agent has 5 actions: 5 Factories or doing noting
        Low level agent has 2 actions: assign or not
        '''
        self.action_space_top = spaces.Discrete(5)
        self.action_space_low = spaces.Discrete(2)
        '''
        Observation space
        '''
        self.observation_space_top = spaces.Box(low=-2,high=2,shape=(9,))
        self.observation_space_low = spaces.Box(low=-2,high=2,shape=(9,))

        # Agent step length
        self.step_length = 1000

        self.done = {}
        self.episode_count = 0
        self.step_num = 0

        with open(self.result_file,'w') as f:
            f_csv = writer(f)
            f_csv.writerow(['time','A','B','P12','P23','current_lorry'])
        with open(self.lorry_file,'w') as f:
            f_csv = writer(f)
            f_csv.writerow(['time','lorry id','MDP','state'])
        with open(self.reward_file,'w') as f:
            f_csv = writer(f)
            f_csv.writerow(['step','reward','cumulate reward'])

    def init_sumo(self):
        # Close existing traci connection
        try:
            traci.close()
            print('restart sumo')
        except:
            pass
        print(f"using {self.num_cpu} cpus")
        traci.start(["sumo", "-c", "/home/lwh/Documents/Code/PredM/"+self.map_file,"--threads",self.num_cpu,"--no-warnings","True"])