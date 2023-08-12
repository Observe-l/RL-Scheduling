import numpy as np
import pandas as pd
import traci
from .core import Truck, Factory, World

class Scenario(object):
    def make_world():
        world = World()

        # add agents, 12 trucks, 4 factories
        num_trucks = 12
        num_factories = 4
        world.agents = [Truck(truck_id='truck_'+str(i)) for i in range(num_trucks)]

        world.agents.append[Factory(factory_id='Factory0', produce_rate=[['P1',5,None,None]])]
        world.agents.append[Factory(factory_id='Factory1', produce_rate=[['P2',10,None,None],['P12',2.5,'P1,P2','1,1']])]
        world.agents.append[Factory(factory_id='Factory2', produce_rate=[['P3',5,None,None],['P23',2.5,'P2,P3','1,1'],['A',2.5,'P12,P3','1,1']])]
        world.agents.append[Factory(factory_id='Factory3', produce_rate=[['P4',5,None,None],['B',2.5,'P23,P4','1,1']])]

        return world()
    
    def reset_world(self, world):
        try:
            traci.close()
            print('restart sumo')
        except:
            pass
        traci.start(["sumo", "-c", "map/3km_1week/osm.sumocfg","--threads",8])
        for agent in world.agents:
            agent.reset()

    def truck_reward(self, agent, world):
        '''
        Calculate reward for the given truck agent.
        The reward depends on the waitting time and the number of product transported during last time step
        '''
        rew = 0
        rew = agent.total_product - agent.last_transport
        return rew
    
    def factory_reward(self, ageng, world):
        '''
        '''
        rew = 0
    
    def truck_agents(self, world):
        return [agent for agent in world.agents if agent.truck]

    def factory_agents(self,world):
        return [agent for agent in world.agents if not agent.truck]
    
    def observation(self, agent, world):
        '''
        The observation of trucks and factories.
        For trucks: 1) Distance to the all factories; 2) Communication (other trucks destination); 3) The state of the trucks
        For factories: the storage of material and product.
        '''

        # The distance between trucks and the destination
        distance = []
        factory_agents = self.factory_agents(world)
        truck_agents = self.truck_agents(world)
        if agent.truck:
            '''
            observation of trucks
            '''
            for factory_agent in factory_agents:
                distance.append(agent.get_distance(factory_agent.id))
            state = agent.get_truck_state()

            # Comunicate with other trucks, get their action.
            com_destination = []
            truck_agents = self.truck_agents(world)
            for other in truck_agents:
                if other is agent: continue
                com_destination.append(other.get_destination)
            return np.concatenate([distance] + [com_destination])
        else:
            '''
            obesrvation of factories
            '''
            product_storage = []
            for factory_product in agent.product.index:
                product_storage.append(agent.contriner.loc[factory_product,'storage'])
            material_storage = []

        


        


    # def set_action(self, actions, agents):
    #     '''
    #     Set action for the agent, both Factories and Trucks.
    #     First, factory take action, and then, trucks take action.
    #     '''
    #     # Set action for factory
    #     for 