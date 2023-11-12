import numpy as np
import pandas as pd
import traci
from .core import Truck, Factory, World, product_management

class Scenario(object):
    def make_world(self):
        # Start Traci
        try:
            traci.close()
            print('restart sumo')
        except:
            pass
        traci.start(["sumo", "-c", "map/3km_1week/osm.sumocfg","--threads","20","--no-warnings","True"])

        world = World()

        # add agents, 12 trucks, 4 factories
        num_trucks = 12
        num_factories = 4
        world.agents = [Truck(truck_id='truck_'+str(i)) for i in range(num_trucks)]

        world.agents.append(Factory(factory_id='Factory0', produce_rate=[['P1',5,None,None]]))
        world.agents.append(Factory(factory_id='Factory1', produce_rate=[['P2',10,None,None],['P12',2.5,'P1,P2','1,1']]))
        world.agents.append(Factory(factory_id='Factory2', produce_rate=[['P3',5,None,None],['P23',2.5,'P2,P3','1,1'],['A',2.5,'P12,P3','1,1']]))
        world.agents.append(Factory(factory_id='Factory3', produce_rate=[['P4',5,None,None],['B',2.5,'P23,P4','1,1']]))

        world.manager = product_management(world.factory_agents(), world.truck_agents())
        for _ in range(300):
            traci.simulationStep()
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in world.truck_agents()]

        return world
    
    def reset_world(self, world) -> None:
        try:
            traci.close()
            print('restart sumo')
        except:
            pass
        traci.start(["sumo-gui", "-c", "map/3km_1week/osm.sumocfg","--threads","20","--no-warnings","True"])
        for agent in world.agents:
            agent.reset()
        
        for _ in range(300):
            traci.simulationStep()
            tmp_state = [tmp_truck.refresh_state() for tmp_truck in world.truck_agents()]

    def reward(self,agent,world) -> float:
        '''
        Reward of both trcuks or factories
        '''
        if agent.truck:
            main_reward = self.truck_reward(agent, world)
        else:
            main_reward = self.factory_reward(agent,world)

        return main_reward

        
    def truck_reward(self, agent, world) -> float:
        '''
        Calculate reward for the given truck agent.
        The reward depends on the waitting time and the number of product transported during last time step
        '''

        # Short-term reward 1: change of transported product in next factory
        rew_1 = 0
        penalty = 0
        for factory_agent in world.factory_agents():
            rew_1 += factory_agent.step_emergency_product[agent.destination]

            # Penalty: going to wrong factory
            if agent.destination == factory_agent.id and factory_agent.req_truck is False:
                penalty = -10

        
        # Short-term reward 2: depends on the distance of between trucks and the destination
        distance = agent.get_distance(agent.destination)
        if distance < 0:
            distance = 0
        # Normalize the distance (min-max scale), assume maximum distance is 5000
        norm_distance = distance / 5000
        rew_2 = -10 * np.log(norm_distance)

        # Get shared Long-term reward
        long_rew = self.shared_reward(world)
        
        rew = rew_1 + rew_2 + penalty  + long_rew
        return rew
    
    def factory_reward(self, agent, world) -> float:
        '''
        Read the reward from factory agent.
        '''
        # Short-term reward 1: change of production num
        rew_1 = 1 * agent.step_transport

        # Short-term reward 2: change of transported product in next factory
        # Penalty: when the factory run out of material
        rew_2 = 0
        peanlty = 0
        for factory_agent in world.factory_agents():
            rew_2 += factory_agent.step_emergency_product[agent.id]
            peanlty += factory_agent.penalty[agent.id]
        # Get shared Long-term reward
        long_rew = self.shared_reward(world)

        rew = rew_1 + rew_2 + long_rew + peanlty
        return rew
    
    def shared_reward(self, world) -> float:
        '''
        Long-term shared reward
        '''
        shared_rew = 0
        rew_trans = 0
        rew_product = 0

        # Reward 1: Total transportated product during last time step
        # P1 = 20
        for truck_agent in world.truck_agents():
            # rewrite, problem
            rew_trans += 20 * truck_agent.total_product - truck_agent.last_transport

        # Reward 2: Number of final product
        # P2 = 20
        for factory_agent in world.factory_agents():
            rew_product += 20 * factory_agent.step_final_product
        

        shared_rew = rew_trans + rew_product
        return shared_rew

    
    def observation(self, agent, world):
        '''
        The observation of trucks and factories.
        For trucks: 1) Distance to the all factories; 2) Communication (other trucks destination); 3) The state of the trucks
        For factories: the storage of material and product.
        '''

        '''
        No communication
        '''
        # The distance between trucks and the destination
        distance = []
        factory_agents = world.factory_agents()
        truck_agents = world.truck_agents()
        if agent.truck:
            '''
            observation of trucks
            '''
            # Communicate with factory
            com_truck_num = []
            tmp_truck_num = 0
            com_factory_action = []

            for factory_agent in factory_agents:
                # Observation 1: distance to 4 factories, [0,+inf]
                tmp_distance = agent.get_distance(factory_agent.id)
                if tmp_distance < 0:
                    tmp_distance = 0
                distance.append(tmp_distance)
                for truck_agent in truck_agents:
                    if truck_agent.destination == factory_agent.id:
                        tmp_truck_num += 1
                # Observation 2: number of trucks that driving to each factory
                com_truck_num.append(tmp_truck_num)
                tmp_truck_num = 0
                # Observation 3: the action of factory agent
                tmp_factory_action = 1 if factory_agent.req_truck is True else 0
                com_factory_action.append(tmp_factory_action)
            # Observation 4: The state of the truck
            state = agent.get_truck_state()

            return np.concatenate([distance] + [com_truck_num] + [com_factory_action] + [[state]])
        
        else:
            '''
            obesrvation of factories
            '''
            # Get the storage of product
            product_storage = []
            for factory_product in agent.product.index:
                product_storage.append(agent.container.loc[factory_product,'storage'])
            # Get the storage of the material
            material_storage = []
            material_index = agent.get_material()
            for tmp_material in material_index:
                material_storage.append(agent.container.loc[tmp_material,'storage'])
            # Get the number of trucks at current factories
            truck_num = 0
            for truck_agent in truck_agents:
                if truck_agent.destination == agent.id:
                    truck_num += 1
            
            return np.concatenate([product_storage] + [material_storage] + [[truck_num]])

