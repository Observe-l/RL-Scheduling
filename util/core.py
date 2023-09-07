import traci
import numpy as np
import pandas as pd
from csv import writer

import random

class Truck(object):
    '''
    The class of truck. 
    Parameters: truck ID, health, position, destination ...
    Function: updata health, move to some positon, fix or broken ...
    '''
    def __init__(self, truck_id:str = 'truck_0', capacity:float = 2.0, weight:float = 0.0,\
                 state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0', product:str = 'A',\
                 path:str = 'result') -> None:
        '''
        Parameters:
        truck_id: string
        capacity: The maximum capacity(t) of truck. Default value is 10 t
        product: current loading product
        weight: Current weight of cargo(kg).
        state: The state of the truck: waitting, loading, pending, delivery, repair, broken, maintenance
        position: string
        destination: string
        path: save the experiments result

        '''
        self.id = truck_id
        self.truck = True

        self.reset(weight,state,position,destination,product)

        self.capacity = capacity

        # sumo time
        self.time_step = 0
        # record total transported product
        self.total_product = 0.0
        self.last_transport = 0.0

        self.path = path + '/truck_record.csv'
    
    def reset(self,weight:float = 0.0, state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0', product:str = 'A'):
        # Create truck in sumo. If the truck already exist, remove it first
        try:
            traci.vehicle.add(vehID=self.id, routeID=position + '_to_'+ destination, typeID='truck')
        except:
            traci.vehicle.remove(vehID=self.id)
            traci.vehicle.add(vehID=self.id, routeID=position + '_to_'+ destination, typeID='truck')
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
        self.last_transport = 0.0

    def update_truck(self, capacity:float = 10000.0, weight:float = 0.0,\
                     state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0') -> None:
        '''
        update the parameters
        '''
        self.capacity = capacity
        self.weight = weight

        self.state = state
        self.position = position
        self.destination = destination


    def refresh_state(self) -> dict:
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
            traci.vehicle.add(vehID=self.id,routeID=self.destination + '_to_'+ self.destination, typeID='truck')
            traci.vehicle.setParkingAreaStop(vehID=self.id,stopID=self.destination)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            tmp_pk = traci.vehicle.getStops(vehID=self.id)
            parking_state = tmp_pk[-1]

        self.position = parking_state.stoppingPlaceID
        
        if parking_state.arrival < 0:
            self.state = 'delivery'
            if len(tmp_pk)>1:
                self.truck_resume()
        elif self.weight == self.capacity and self.position == self.destination:
            self.state = 'pending for unloading'
        elif self.weight == 0:
            self.state = 'waitting'

        return {'state':self.state, 'postion':self.position}


    def truck_stop(self) -> None:
        '''
        When truck broken or we decide to repair / maintain the truck,
        use this function to stop the truck first
        Time-based function
        '''
        # The truck shouldn't break at factory road, otherwise, let it move to the end of the road
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
    
    def truck_resume(self) -> None:
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
        # Remove vehicle first, add another truck. (If we want to use the dijkstra algorithm in SUMO, we must creat new vehicle)
        self.destination = destination
        traci.vehicle.changeTarget(vehID=self.id, edgeID=destination)
        # Move out the car parking area
        traci.vehicle.resume(vehID=self.id)
        # Stop at next parking area
        traci.vehicle.setParkingAreaStop(vehID=self.id, stopID=self.destination)
        #print(f'[move] {self.id} move from {self.position} to {self.destination}')

    def load_cargo(self, weight:float, product:str) -> tuple:
        '''
        Load cargo to the truck. Cannot exceed the maximum capacity. The unit should be 'kg'.
        After the truck is full, the state will change to pending, and the color change to Red
        If truck is not empty, it would be blue color
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
            # After the truck is full, record it
            self.last_transport = self.total_product
            self.total_product += self.weight
            return ('full', self.weight + weight - self.capacity)
    
    def unload_cargo(self, weight:float) -> tuple:
        '''
        Unload cargo. If truck is empty, health become waitting.
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
    
    def get_distance(self, positon) -> float:
        traci.vehicle.changeTarget(vehID=self.id, edgeID=positon)
        distance = traci.vehicle.getDrivingDistance(vehID=self.id, edgeID=positon, pos=0)
        traci.vehicle.changeTarget(vehID=self.id, edgeID=self.destination)

        return distance
    
    def get_truck_state(self) -> int:
        if self.state == 'waitting':
            return 1
        else:
            return 0
    
    def get_destination(self) -> int:
        truck_destination = int(self.destination[-1])
        return truck_destination
        


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
        self.truck = False

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

        # There two dimension of action: 1) Need trucks or not; 2) The number of new trucks
        self.req_truck = False
        self.req_num = 0
        # The number of trucks which desitination is current factory or stop at current factory
        self.truck_num = 0

        self.step = 0

    def reset(self) -> None:
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
    
    def load_cargo(self, truck:Truck, product:str) -> str:
        '''
        Load cargo to the truck in current factory
        '''
        # Check the state and position of the truck
        # Check the storage
        if self.id in truck.position and (truck.state == 'waitting' or truck.state == 'loading') and self.container.loc[product,'storage'] != 0:
            # if truck.state == 'waitting':
                # print when startting loading
                # print(f'[loading] {truck.id} start loading {product} at:{self.id}')
            # Maximum loading speed: 0.05 t/s
            load_weight = min(0.05, self.container.loc[product,'storage'])
            truck_state, exceed_cargo =  truck.load_cargo(weight=load_weight, product= product)
            self.container.at[product,'storage'] = self.container.loc[product,'storage'] - (load_weight - exceed_cargo)
            return truck_state
    
    def unload_cargo(self, truck:Truck) -> None:
        '''
        Unload cargo to container
        '''
        if self.id in truck.position and (truck.state == 'pending for unloading' or truck.state == 'unloading') and self.container.loc[truck.product,'storage'] < self.container.loc[truck.product,'capacity']:
            # if truck.state == 'pending for unloading':
                # print when startting unloading
                # print(f'[unloading] {truck.id} start unloading {truck.product} at:{self.id}')
            # Maximum loading speed: 0.05 t/s
            unload_weight = min(0.05, self.container.loc[truck.product,'capacity'] - self.container.loc[truck.product,'storage'])
            truck_state, exceed_cargo = truck.unload_cargo(unload_weight)
            self.container.at[truck.product,'storage'] = self.container.loc[truck.product,'storage'] + (unload_weight - exceed_cargo)
        
    def get_material(self) -> list:
        tmp_material = list(filter(lambda item: item is not None,self.product['material'].values.tolist()))
        material = []
        for i in tmp_material:
            i = i.split(',')
            material.extend(i)
        return material

class product_management(object):
    '''
    product new product, load cargo, etc.
    '''
    def __init__(self, factory:list, truck:list) -> None:
        '''
        Input the list of factories and the trucks
        Producding order:
        Factory0: produce P1
        Facotry1: produce P2, P12
        Factory2: produce P3, P23, A(P123)
        Factory3: produce P4, B(P234)
        '''
        self.factory = factory
        self.truck = truck
        self.p = np.array([1.0, 1.0, 1.0, 1.0, truck[0].capacity])
        # Create the dictionary for product
        # self.product_idx = {tmp_factory.id:tmp_factory.product.index.values.tolist() for tmp_factory in self.factory}
        self.product_idx = {'Factory0':['P1'],'Factory1':['P12','P2'],'Factory2':['P23'],'Factory3':[]}
        self.transport_idx = {'P1':'Factory1',
                              'P2':'Factory2','P12':'Factory2',
                              'P23':'Factory3'}
    
    def produce_load(self) -> None:
        '''
        Produce new product in all factories
        '''

        for tmp_factory in self.factory:
            tmp_factory.produce_product()
            for tmp_truck in self.truck:
                tmp_factory.unload_cargo(tmp_truck)
            # Start loading the product to truck.
            # Only when the product is enough to full the truck
            tmp_product = self.product_idx[tmp_factory.id]
            truck_pool = [truck for truck in self.truck if truck.position == tmp_factory.id and truck.state == 'waitting']

            # Continue loading
            truck_continue = [truck for truck in self.truck if truck.position == tmp_factory.id and truck.state == 'loading']
            for tmp_truck in truck_continue:
                if tmp_truck.position == tmp_factory.id:
                    tmp_result = tmp_factory.load_cargo(tmp_truck,tmp_truck.product)
                    if tmp_result == 'full':
                        # print(f'[delievery] {tmp_truck.id} delivers the {tmp_truck.product}')
                        tmp_truck.delivery(self.transport_idx[tmp_truck.product])
            
            # for item in tmp_product:
                # print(item not in truck_duplicate)
            truck_duplicate = [truck.product for truck in self.truck if truck.position == tmp_factory.id and truck.state == 'loading']
            if len(tmp_product) == 2:
                # loading the product with max storage
                item = self.factory[2].container.loc[tmp_product,'storage'].idxmin()
                item_bak = [i for i in tmp_product if i != item][0]
                if (tmp_factory.container.loc[item,'storage'] > self.truck[0].capacity) and (item not in truck_duplicate) and (len(truck_pool)>0):
                    tmp_result = tmp_factory.load_cargo(truck_pool[0],item)
                elif (tmp_factory.container.loc[item_bak,'storage'] > self.truck[0].capacity) and (item_bak not in truck_duplicate) and (len(truck_pool)>0):
                    tmp_result = tmp_factory.load_cargo(truck_pool[0],item_bak)

            elif len(tmp_product) == 1:
                item = tmp_product[0]
                if (tmp_factory.container.loc[item,'storage'] > self.truck[0].capacity) and (item not in truck_duplicate) and (len(truck_pool)>0):
                    tmp_result = tmp_factory.load_cargo(truck_pool[0],item)

class World(object):
    def __init__(self):
        # list of agents 
        self.agents = []
        # factory load/product manager
        self.manager = []
        
    
    def step(self):
        '''
        update the world
        '''
        traci.simulationStep()
        for agent in self.agents:
            # Refresh the state of the truck
            if agent.truck:
                agent.refresh_state()

        self.manager.produce_load()