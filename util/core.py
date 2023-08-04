import traci
import numpy as np
import pandas as pd
from csv import writer

import random

class Lorry(object):
    '''
    The class of lorry. 
    Parameters: lorry ID, health, position, destination ...
    Function: updata health, move to some positon, fix or broken ...
    '''
    def __init__(self, lorry_id:str = 'lorry_0', capacity:float = 2.0, weight:float = 0.0,\
                 state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0', product:str = 'A',\
                 path:str = 'result') -> None:
        '''
        Parameters:
        lorry_id: string
        capacity: The maximum capacity(t) of lorry. Default value is 10 t
        product: current loading product
        weight: Current weight of cargo(kg).
        state: The state of the lorry: waitting, loading, pending, delivery, repair, broken, maintenance
        position: string
        destination: string
        path: save the experiments result

        '''
        self.id = lorry_id
        self.reset(weight,state,position,destination,product)

        self.capacity = capacity

        # sumo time
        self.time_step = 0
        # record total transported product
        self.total_product = 0.0

        self.path = path + '/lorry_record.csv'
    
    def reset(self,weight:float = 0.0, state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0', product:str = 'A'):
        # Create lorry in sumo. If the lorry already exist, remove it first
        try:
            traci.vehicle.add(vehID=self.id, routeID=position + '_to_'+ destination, typeID='lorry')
        except:
            traci.vehicle.remove(vehID=self.id)
            traci.vehicle.add(vehID=self.id, routeID=position + '_to_'+ destination, typeID='lorry')
        traci.vehicle.setParkingAreaStop(vehID=self.id,stopID=position)

        self.weight = weight
        self.state = state
        self.position = position
        self.destination = destination
        self.product = product
        self.color = (255,255,0,255)
        self.recover_state = 'waitting'

        # record total transported product
        self.total_product = 0.0

    def update_lorry(self, capacity:float = 10000.0, weight:float = 0.0,\
                     state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0') -> None:
        '''
        update the parameters
        '''
        self.capacity = capacity
        self.weight = weight

        self.state = state
        self.position = position
        self.destination = destination


    def refresh_state(self,time_step, repair_flag) -> dict:
        '''
        get current state, refresh state
        '''

        # Check current location, if the vehicle remove by SUMO, add it first
        try:
            tmp_pk = traci.vehicle.getStops(vehID=self.id)
            parking_state = tmp_pk[-1]
        except:
            try:
                # print(f'{self.id}, position:{self.position}, destination:{self.destination}, parking: {traci.vehicle.getStops(vehID=self.id)}, state: {self.state}')
                # print(f'weight: {self.weight}, mdp state: {self.mk_state}')
                traci.vehicle.remove(vehID=self.id)
            except:
                # print(f'{self.id} has been deleted')
                # print(f'weight: {self.weight}, mdp state: {self.mk_state}')
                pass
            traci.vehicle.add(vehID=self.id,routeID=self.destination + '_to_'+ self.destination, typeID='lorry')
            traci.vehicle.setParkingAreaStop(vehID=self.id,stopID=self.destination)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            tmp_pk = traci.vehicle.getStops(vehID=self.id)
            parking_state = tmp_pk[-1]

        self.position = parking_state.stoppingPlaceID
        
        if parking_state.arrival < 0:
            self.state = 'delivery'
            if len(tmp_pk)>1:
                self.lorry_resume()
        elif self.weight == self.capacity and self.position == self.destination:
            self.state = 'pending for unloading'
        elif self.weight == 0:
            self.state = 'waitting'

        return {'state':self.state, 'postion':self.position}


    def lorry_stop(self):
        '''
        When lorry broken or we decide to repair / maintain the lorry,
        use this function to stop the lorry first
        Time-based function
        '''
        # The lorry shouldn't break at factory road, otherwise, let it move to the end of the road
        current_edge = traci.vehicle.getRoadID(vehID=self.id)
        factory_idx = ['Factory0','Factory1','Factory2','Factory3']
        # arrive the destination
        if self.destination == current_edge:
            if self.weight == 0:
                self.recover_state = 'waitting'
            else:
                self.recover_state = 'pending for unloading'
        # start from current factory
        elif current_edge in factory_idx:
            self.recover_state = 'delivery'
            try:
                # stop after 20 meters barking
                traci.vehicle.setStop(vehID=self.id,edgeID=traci.vehicle.getRoadID(vehID=self.id),pos=150)
            except:
                # stop at next edge. the length of the edge must longer than 25m
                tmp_idx = traci.vehicle.getRouteIndex(vehID=self.id)
                tmp_edge = traci.vehicle.getRoute(vehID=self.id)[tmp_idx+2]
                traci.vehicle.setStop(vehID=self.id,edgeID=tmp_edge,pos=0)
        else:
            self.recover_state = 'delivery'
            try:
                # stop after 20 meters barking
                traci.vehicle.setStop(vehID=self.id,edgeID=traci.vehicle.getRoadID(vehID=self.id),pos=traci.vehicle.getLanePosition(vehID=self.id)+25)
            except:
                # stop at next edge. the length of the edge must longer than 25m
                tmp_idx = traci.vehicle.getRouteIndex(vehID=self.id)
                try:
                    tmp_edge = traci.vehicle.getRoute(vehID=self.id)[tmp_idx+1]
                    traci.vehicle.setStop(vehID=self.id,edgeID=tmp_edge,pos=25)
                except:
                    if self.weight == 0:
                        self.recover_state = 'waitting'
                    else:
                        self.recover_state = 'pending for unloading'
    
    def lorry_resume(self):
        tmp_pk = traci.vehicle.getStops(vehID=self.id)
        if len(tmp_pk) > 0:
            latest_pk = tmp_pk[0]
            if latest_pk.arrival > 0:
                traci.vehicle.resume(vehID=self.id)


    def delivery(self, destination:str) -> None:
        '''
        delevery the cargo to another factory
        '''
        self.state = 'delivery'
        # Remove vehicle first, add another lorry. (If we want to use the dijkstra algorithm in SUMO, we must creat new vehicle)
        self.destination = destination
        traci.vehicle.changeTarget(vehID=self.id, edgeID=destination)
        # Move out the car parking area
        traci.vehicle.resume(vehID=self.id)
        # Stop at next parking area
        traci.vehicle.setParkingAreaStop(vehID=self.id, stopID=self.destination)
        #print(f'[move] {self.id} move from {self.position} to {self.destination}')

    def load_cargo(self, weight:float, product:str):
        '''
        Load cargo to the lorry. Cannot exceed the maximum capacity. The unit should be 'kg'.
        After the lorry is full, the state will change to pending, and the color change to Red
        If lorry is not empty, it would be blue color
        '''
        self.product = product
        if self.weight + weight < self.capacity:
            self.weight += weight
            self.state = 'loading'
            # RGBA
            self.color = (0,0,255,255)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            return ('successful', 0.0)
        else:
            self.weight = self.capacity
            self.state = 'pending for delivery'
            self.color = (255,0,0,255)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            # After the lorry is full, record it
            self.total_product += self.weight
            return ('full', self.weight + weight - self.capacity)
    
    def unload_cargo(self, weight:float):
        '''
        Unload cargo. If lorry is empty, health become waitting.
        '''
        self.state = 'unloading'
        self.color = (0,0,255,255)
        traci.vehicle.setColor(typeID=self.id,color=self.color)
        if weight <= self.weight:
            self.weight -= weight
            return ('successful', 0.0)
        else:
            remainning_weight = self.weight
            self.weight =0
            self.state = 'waitting'
            self.color = (0,255,0,255)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            return ('not enough', remainning_weight)


class Factory(object):
    '''
    The class of factory
    '''
    def __init__(self, factory_id:str = 'Factory0', produce_rate:list = [['P1',0.0001,None,None]], 
                 capacity:float=800.0, container:list = ['P1','P2','P3','P4','P12','P23','A','B']) -> None:
        '''
        Parameters:
        factory_id: string
        produce_rate: 2d list. Sample: [[product name (string), produce rate(float), required materials(string, using ',' to split multiple materials), ratio(string)]]
        capacity: list. Volume of the containers
        container: list of container, the element is product name.
        '''
        self.id = factory_id
        self.produce_rate = produce_rate
        self.container = container

        # Create a dataframe to record the products which are produced in current factory
        self.product= pd.DataFrame(self.produce_rate,columns=['product','rate','material','ratio'])
        self.product['total'] = [0.0] * len(self.produce_rate)
        self.product.set_index(['product'],inplace=True)
        # The dataframe of the container
        self.container = pd.DataFrame({'product':self.container, 'storage':[0.0]*len(self.container), 'capacity':[capacity] * 4 + [60000] * 4})
        self.container.set_index(['product'],inplace=True)
        self.container.at['P2','capacity'] = 2*capacity
        self.reset()

        self.step = 0

    def reset(self):
        '''
        Set total storage and total producd to 0
        '''
        self.product['total'] = [0.0] * len(self.produce_rate)
        self.container['storage'] = [0.0]*len(self.container)

    
    def produce_product(self) -> None:
        '''
        Produce new product. Won't exceed container capacity
        '''
        # Iterate all the product
        for index, row in self.product.iterrows():
            # Check the materials in the container
            tmp_rate = row['rate']
            # Storage shouldn't exceed capacity
            item_num = min(tmp_rate,self.container.loc[index,'capacity'] - self.container.loc[index,'storage'])
            item_num = max(item_num, 0)
            tmp_materials = row['material']
            if type(tmp_materials) == str:
                tmp_materials = tmp_materials.split(',')
                tmp_ratio = np.array(row['ratio'].split(','),dtype=np.float64)

                tmp_storage = self.container.loc[tmp_materials,'storage'].to_numpy()
                # Check storage
                if (tmp_storage > tmp_ratio*tmp_rate).all() and self.container.loc[index,'capacity'] > self.container.loc[index,'storage']:
                    # Consume the material
                    for i in range(len(tmp_materials)):
                        self.container.at[tmp_materials[i],'storage'] = self.container.loc[tmp_materials[i],'storage'] - item_num * tmp_ratio[i]
                    # Produce new product
                    self.container.at[index,'storage'] = self.container.loc[index,'storage'] + item_num
                    self.product.at[index,'total'] = self.product.loc[index,'total'] + item_num

            # no need any materials
            else:
                # Produce directly
                self.container.at[index,'storage'] = self.container.loc[index,'storage'] + item_num
                self.product.at[index,'total'] = self.product.loc[index,'total'] + item_num
    
    def load_cargo(self, lorry:Lorry, product:str) -> str:
        '''
        Load cargo to the lorry in current factory
        '''
        # Check the state and position of the lorry
        # Check the storage
        if self.id in lorry.position and (lorry.state == 'waitting' or lorry.state == 'loading') and self.container.loc[product,'storage'] != 0:
            # if lorry.state == 'waitting':
                # print when startting loading
                # print(f'[loading] {lorry.id} start loading {product} at:{self.id}')
            # Maximum loading speed: 0.05 t/s
            load_weight = min(0.05, self.container.loc[product,'storage'])
            lorry_state, exceed_cargo =  lorry.load_cargo(weight=load_weight, product= product)
            self.container.at[product,'storage'] = self.container.loc[product,'storage'] - (load_weight - exceed_cargo)
            return lorry_state
    
    def unload_cargo(self, lorry:Lorry) -> None:
        '''
        Unload cargo to container
        '''
        if self.id in lorry.position and (lorry.state == 'pending for unloading' or lorry.state == 'unloading') and self.container.loc[lorry.product,'storage'] < self.container.loc[lorry.product,'capacity']:
            # if lorry.state == 'pending for unloading':
                # print when startting unloading
                # print(f'[unloading] {lorry.id} start unloading {lorry.product} at:{self.id}')
            # Maximum loading speed: 0.05 t/s
            unload_weight = min(0.05, self.container.loc[lorry.product,'capacity'] - self.container.loc[lorry.product,'storage'])
            lorry_state, exceed_cargo = lorry.unload_cargo(unload_weight)
            self.container.at[lorry.product,'storage'] = self.container.loc[lorry.product,'storage'] + (unload_weight - exceed_cargo)


class World(object):
    def __init__(self):
        # list of agents 
        self.agents = []
        
    
    def step(self):
        '''
        update the 
        '''
        self.agents = []
        