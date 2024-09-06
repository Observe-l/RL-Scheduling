import gymnasium as gym
# import traci
import libsumo as traci
import numpy as np
import random
from gymnasium.spaces import Discrete, Box
from csv import writer
from pathlib import Path
from .core import Truck, Factory, product_management
import string
import xml.etree.ElementTree as ET
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class async_scheduling(MultiAgentEnv):
    def __init__(self, env_config):
        # 12 Trucks, 4 Factories. The last factory is not agent
        self.truck_num = 12
        self.factory_num = 50
        # init sumo at the begining
        self.init_sumo()
        # Define the observation space and action space.
        self.observation_space = {}
        self.action_space = {}
        obs = self._get_obs()
        for agent_id, tmp_obs in obs.items():
            obs_dim = len(tmp_obs)
            self.observation_space[agent_id] = Box(low=0, high=+np.inf, shape=(obs_dim,),dtype=np.float32)
            self.action_space[agent_id] = Discrete(self.factory_num)
        # The done flag
        self.done = {}

        self.episode_num = 0
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.path = f"/home/lwh/Documents/Code/RL-Scheduling/result/{env_config['algo']}/exp_{random_string}"

    def reset(self):
        '''
        Reset the environment state and return the initial observations for each agent.
        '''
        # Create folder and file to save the data
        self.make_folder()
        # Count the episode num
        self.episode_num += 1
        # Init SUMO
        self.init_sumo()
        # Get init observation
        obs = self._get_obs()
        # Operable penalty
        self.operable_penalty = {}
        # done flag
        self.done['__all__'] = False
        return obs

    def step(self, action_dict):
        '''
        Compute the environment dynamics given the actions of each agent.
        Return a dictionary of observations, rewards, dones (indicating whether the episode is finished), and info.
        '''
        # Set action
        self._set_action(action_dict)
        # Run SUMO until all the agents are avaiable
        sumo_flag = True
        # Record step lenth
        step_lenth = 0
        # The SUMO simulation
        while sumo_flag:
            traci.simulationStep()
            # Refresh truck state
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in self.truck_agents]
            self.manager.rl_produce_load()
            trucks_operable = [tmp_truck.operable_flag for tmp_truck in self.truck_agents]
            # If any of the trucks are operable, break the loop
            sumo_flag = False if any(trucks_operable) else True
            step_lenth += 1
        
        # Resume all trucks to get observation
        self.resume_truck()
        obs = self._get_obs()
        rewards = self._get_reward(action_dict.keys())
        # Park all truck to continue the simulation
        self.park_truck()
        # Reset the flag
        self.flag_reset()
        # Save the results
        current_time = round(traci.simulation.getTime() / 3600, 3)
        with open(self.product_file, 'a') as f:
            f_csv = writer(f)
            tmp_A = round(self.factory[45].product.loc['A','total'],3)
            tmp_B = round(self.factory[46].product.loc['B','total'],3)
            tmp_C = round(self.factory[47].product.loc['C','total'],3)
            tmp_D = round(self.factory[48].product.loc['D','total'],3)
            tmp_E = round(self.factory[49].product.loc['E','total'],3)
            total = tmp_A+tmp_B+tmp_C+tmp_D+tmp_E
            product_list = [current_time,step_lenth,total,tmp_A,tmp_B,tmp_C,tmp_D,tmp_E]
            f_csv.writerow(product_list)
        
        with open(self.agent_file, 'a') as f:
            f_csv = writer(f)
            agent_list = [current_time, step_lenth]
            for tmp_agent in self.truck_agents:
                agent_id = int(tmp_agent.id.split('_')[1])
                if agent_id in action_dict.keys():
                    tmp_agent.cumulate_reward += rewards[agent_id]
                    agent_list += [action_dict[agent_id], rewards[agent_id], tmp_agent.cumulate_reward]
                else:
                    agent_list += ['NA', 'NA', tmp_agent.cumulate_reward]
            f_csv.writerow(agent_list)
        
        with open(self.distance_file, 'a') as f:
            f_csv = writer(f)
            distance_list = [current_time]
            for tmp_agent in self.truck_agents:
                distance_list += [tmp_agent.step_distance, tmp_agent.driving_distance]
            f_csv.writerow(distance_list)
        
        if current_time >= 3600*24:
            self.done['__all__'] = True

        return obs, rewards, self.done, {}

    def _get_obs(self) -> dict:
        '''
        Return back a dictionary for operable agents
        '''
        observation = {}
        # Shared observation, Storage/Queue from factory
        product_storage = []
        material_storage = []
        com_truck_num = []
        for factory_agent in self.factory:
            # Get the storage of product
            for factory_product in factory_agent.product.index:
                product_storage.append(factory_agent.container.loc[factory_product,'storage'])
            # Get the storage of the material
            material_index = factory_agent.get_material()
            for tmp_material in material_index:
                material_storage.append(factory_agent.container.loc[tmp_material,'storage'])
            # Get the number of trucks at current factories
            tmp_truck_num = 0
            for tmp_truck in self.truck_agents:
                if tmp_truck.destination == factory_agent.id:
                    tmp_truck_num += 1
            com_truck_num.append(tmp_truck_num)
        queue_obs = np.concatenate([product_storage] +[material_storage] + [com_truck_num])

        # The truck agents' observation. Only use the truck that operable.
        operable_trucks = [tmp_truck for tmp_truck in self.truck_agents if tmp_truck.operable_flag]
        for truck_agent in operable_trucks:
        # for truck_agent, agent_id in zip(self.truck_agents,range(len(self.truck_agents))):
            distance = []
            # Distance to 45 factories, [0,+inf]
            for factory_agent in self.factory[:45]:
                tmp_distance = truck_agent.get_distance(factory_agent.id)
                if tmp_distance < 0:
                    tmp_distance = 0
                distance.append(tmp_distance)
            # Current destination
            destination = int(truck_agent.destination[-1])
            # The state of the truck
            state = truck_agent.get_truck_state()
            # The transported product
            product = truck_agent.get_truck_produce()
            agent_id = int(truck_agent.id.split('_')[1])
            observation[agent_id] = np.concatenate([queue_obs] + [distance] + [[destination]] + [[product]] + [[state]])
        
        return observation
    
    def _set_action(self, actions:dict):
        '''
        Set action for all the agent
        '''
        for agent_id, action in actions.items():
            agent = self.truck_agents[agent_id]
            target_id = self.factory[action].id
            # Assign truck to the new destination
            if agent.operable_flag:
                agent.delivery(destination=target_id)
            else:
                pass
    
    def _get_reward(self, act_keys) -> dict:
        '''
        Get reward for given agents
        '''
        rew = {}
        for agent_id in act_keys:
            tmp_agent = self.truck_agents[agent_id]
            rew[agent_id] = self.truck_reward(tmp_agent)
        return rew

    def truck_reward(self, agent) -> float:
        '''
        Calculate reward for the given truck agent.
        The reward depends on the waitting time and the number of product transported during last time step
        '''
        # First factor: unit profile
        rew_final_product = 0
        for factory in self.factory:
            rew_final_product += 40 * factory.step_final_product
        # Second factor: driving cost
        gk = 0.001
        fk = 0.002
        if agent.weight == 0:
            uk = gk+fk
        else:
            uk = gk
        rew_driving = uk * agent.step_distance

        # Third factor: asset cost
        rew_ass = 10

        # Penalty factor
        gamma1 = 0.5
        gamma2 = 0.5
        rq = 1
        tq = 5000 if agent.time_step >= 5000 else agent.time_step
        sq = gamma1 * tq/5000 + gamma2 * (1-rq)
        psq = np.log((1-sq)/(1+sq))

        # Short-term reward. Arrive right factory
        rew_short = agent.last_transport
        if rew_short != 0:
            print(f"{agent.id} transport {rew_short} {agent.product}")

        rew = rew_final_product + rew_short - rew_driving - rew_ass - psq

        # # Reward 1: final product * 40

        # # Reward 2: Transported component durning last time step
        # rew_last_components = agent.last_transport
        
        # # Reward 3: depends on the distance of between trucks and the destination 0~8
        # distance = agent.get_distance(agent.destination)
        # if distance < 0:
        #     distance = 1
        # # Normalize the distance (min-max scale), assume maximum distance is 5000
        # norm_distance = distance / 5000
        # distance_reward = -3 * np.log(norm_distance)

        # rew = rew_final_product + rew_last_components + distance_reward
        return rew

    def init_sumo(self):
        try:
            traci.close()
            print('restart sumo')
        except:
            pass
        traci.start(["sumo", "-c", "map/sg_map/osm.sumocfg","--no-warnings","True"])
        # Get the lane id
        parking_xml = "map/sg_map/factory.prk.xml"
        parking_dict = self.xml_dict(parking_xml)

        self.truck_agents = [Truck(truck_id='truck_'+str(i), factory_edge=parking_dict) for i in range(self.truck_num)]
        # Add factory 0 ~ 44
        self.factory = [Factory(factory_id=f'Factory{i}',produce_rate=[[f'P{i}',5,None,None]]) for i in range(45)]
        # Add factory 45 ~ 49
        '''
        Random raw material list. Each final product has 9 raw materials.
        '''
        final_products = ['A', 'B', 'C', 'D', 'E']
        remaining_materials = [f'P{i}' for i in range(45)]
        transport_idx = {}
        random.shuffle(remaining_materials)
        for i, product in enumerate(final_products):
            tmp_factory_id = f'Factory{45 + i}'
            tmp_materials = [remaining_materials.pop() for _ in range(9)]
            tmp_produce_rate = [[product, 5, ','.join(tmp_materials), ','.join(['1']*len(tmp_materials))]]
            tmp_factory = Factory(factory_id=tmp_factory_id, produce_rate=tmp_produce_rate)
            self.factory.append(tmp_factory)
            for transport_material in tmp_materials:
                transport_idx[transport_material] = tmp_factory_id

        '''
        Fix material list
        '''
        # final_product_details = {
        #     'A': (['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'], 'Factory45'),
        #     'B': (['P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18'], 'Factory46'),
        #     'C': (['P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27'], 'Factory47'),
        #     'D': (['P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36'], 'Factory48'),
        #     'E': (['P37', 'P38', 'P39', 'P40', 'P41', 'P42', 'P43', 'P44', 'P45'], 'Factory49'),
        # }

        self.manager = product_management(self.factory, self.truck_agents, transport_idx)
        for _ in range(100):
            traci.simulationStep()
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in self.truck_agents]
            # self.manager.produce_load()

    def make_folder(self):
        '''
        Create folder to save the result
        '''
        # Create folder
        folder_path = self.path + '/{}/'.format(self.episode_num)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Create file
        self.product_file = folder_path + 'product.csv'
        self.agent_file = folder_path + 'result.csv'
        self.distance_file = folder_path + 'distance.csv'
        # Create result file
        with open(self.product_file,'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length', 'total', 'A', 'B', 'C', 'D', 'E']
            f_csv.writerow(result_list)
        
        with open(self.agent_file, 'w') as f:
            f_csv = writer(f)
            result_list = ['time', 'step_length']
            for agent in self.truck_agents:
                agent_list = ['action_'+agent.id,'reward_'+agent.id,'cumulate reward_'+agent.id]
                result_list += agent_list
            f_csv.writerow(result_list)

        # Create active truck file
        with open(self.distance_file,'w') as f:
            f_csv = writer(f)
            distance_list = ['time']
            for agent in self.truck_agents:
                agent_list = [f'step_{agent.id}', f'total_{agent.id}']
                distance_list += agent_list
            f_csv.writerow(distance_list)
    
    def xml_dict(self, xml_file) -> dict:
        '''
        Get data from xml file.
        Use in the traci api
        '''
        tree = ET.parse(xml_file)
        root = tree.getroot()
        parking_dict = {}
        for parking_area in root.findall('parkingArea'):
            parking_id = parking_area.get('id')
            lane = parking_area.get('lane')
            if lane.endswith('_0'):
                lane = lane[:-2]
            parking_dict[parking_id] = lane
        return parking_dict

    def resume_truck(self):
        '''
        resume all truck from parking area to get the distance
        '''
        for agent in self.truck_agents:
            tmp_pk = traci.vehicle.getStops(vehID=agent.id)
            if len(tmp_pk) > 0:
                latest_pk = tmp_pk[0]
                if latest_pk.arrival > 0:
                    traci.vehicle.resume(vehID=agent.id)
        traci.simulationStep()
    
    def park_truck(self):
        '''
        put all truck back to the parking area
        '''
        for agent in self.truck_agents:
            try:
                traci.vehicle.setParkingAreaStop(vehID=agent.id, stopID=agent.destination)
            except:
                pass
        for _ in range(5):
            traci.simulationStep()
    
    def flag_reset(self):
        for factory_agent in self.factory:
            # The number of pruduced component during last time step
            factory_agent.step_produced_num = 0
            factory_agent.step_final_product = 0
            # The number of decreased component during last time step
            factory_agent.step_transport = 0
            # factory_agent.step_emergency_product = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}
        for truck in self.truck_agents:
            # Reset the number of transported goods
            truck.last_transport = 0

    def stop_env(self):
        traci.close()
