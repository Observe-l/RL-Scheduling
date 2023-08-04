import numpy as np
import pandas as pd
import traci
from .core import Lorry, Factory, World

class Scenario(object):
    def make_world():
        world = World()

        # add agents, 12 trucks, 4 factories
        num_trucks = 12
        num_factories = 4
        world.agents = [Lorry(lorry_id='lorry_'+str(i)) for i in range(num_trucks)]

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

    def reward(self, agent, world):
        '''
        
        '''
        a= 10