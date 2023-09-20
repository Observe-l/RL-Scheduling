import numpy as np
import pandas as pd
import traci
from csv import writer
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
        traci.start(["sumo", "-c", "map/3km_1week/osm.sumocfg","--threads","20","--no-warnings","True"])
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
        
        # Save reward
        with open(agent.id+'.txt','a') as f:
            f_csv = writer(f)
            f_csv.writerow([traci.simulation.getTime(),main_reward])

        return main_reward

        
    def truck_reward(self, agent, world) -> float:
        '''
        Calculate reward for the given truck agent.
        The reward depends on the waitting time and the number of product transported during last time step
        '''
        rew = agent.total_product - agent.last_transport

        # Check the state of the factory
        factory_agents = world.factory_agents()
        for factory_agent in factory_agents:
            if agent.destination == factory_agent.id:
                if factory_agent.req_truck is False:
                    rew -= 10
                break

        return rew
    
    def factory_reward(self, agent, world) -> float:
        '''
        Read the reward from factory agent.
        Set it to 0 after reading.
        '''
        rew = agent.step_produced_num
        agent.step_produced_num = 0
        return rew
    
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
            com_factory = []

            for factory_agent in factory_agents:
                distance.append(agent.get_distance(factory_agent.id))
                tmp_req = 1 if factory_agent.req_truck is True else 0
                com_factory.append(com_factory)
            state = agent.get_truck_state()

            # Comunicate with other trucks, get their action.
            com_destination = []
            # truck_agents = world.truck_agents()
            for other in truck_agents:
                if other is agent: continue
                com_destination.append(other.get_destination())

            return np.concatenate([distance] + [com_destination]+[[com_factory]])
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

