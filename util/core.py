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
    def __init__(self, truck_id:str = 'truck_0', capacity:float = 5.0, weight:float = 0.0,\
                 state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0', product:str = 'A') -> None:
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
        # RGBA Green
        self.color = (255,255,0,255)
        # The operable flag
        self.operable_flag = False
        self.recover_state = 'waitting'

        # record total transported product
        self.total_product = 0.0
        self.last_transport = 0.0
        # sumo time
        self.time_step = 0
        # Record the reward
        self.cumulate_reward = 0.0

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
        self.operable_flag = False
        
        if parking_state.arrival < 0:
            self.state = 'delivery'
            if len(tmp_pk)>1:
                self.truck_resume()
        elif self.weight == self.capacity and self.position == self.destination:
            self.state = 'pending for unloading'
        elif self.weight == 0:
            self.state = 'waitting'
            self.operable_flag = True

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
        self.operable_flag = False
        # Remove vehicle first, add another truck. (If we want to use the dijkstra algorithm in SUMO, we must creat new vehicle)
        self.destination = destination
        traci.vehicle.changeTarget(vehID=self.id, edgeID=destination)
        # Move out the car parking area
        try:
            traci.vehicle.resume(vehID=self.id)
        except:
            pass
        # Stop at next parking area
        try:
            traci.vehicle.setParkingAreaStop(vehID=self.id, stopID=self.destination)
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
            # RGBA Blue
            self.color = (0,0,255,255)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            return ('successful', 0.0)
        else:
            self.weight = self.capacity
            self.state = 'pending for delivery'
            # RGBA Red
            self.color = (255,0,0,255)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            # After the truck is full, record it
            self.total_product += self.weight
            return ('full', self.weight + weight - self.capacity)
    
    def unload_cargo(self, weight:float) -> tuple:
        '''
        Unload cargo. If truck is empty, health become waitting.
        '''
        if weight < self.weight:
            self.state = 'unloading'
            # RGBA Blue
            self.color = (0,0,255,255)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            self.weight -= weight
            self.last_transport += weight
            return ('successful', 0.0)
        else:
            remainning_weight = self.weight
            self.weight =0
            self.state = 'waitting'
            # RGBA Green
            self.color = (255,255,0,255)
            traci.vehicle.setColor(typeID=self.id,color=self.color)
            self.last_transport += remainning_weight
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
                 container:list = ['P1','P2','P3','P4','P12','P23','A','B'],
                 product_source:dict = {'P1':'Factory0',
                                        'P2':'Factory1', 'P12':'Factory1',
                                        'P3':'Factory2', 'P23':'Factory2','A':'Factory2',
                                        'P4':'Factory3', 'B':'Factory3'}) -> None:
        '''
        Parameters:
        factory_id: string
        produce_rate: 2d list. Sample: [[product name (string), produce rate(float), required materials(string, using ',' to split multiple materials), ratio(string)]]
        capacity: list. Volume of the containers
        container: list of container, the element is product name.
        '''
        self.id = factory_id
        self.truck = False

        # Create a dataframe to record the products which are produced in current factory
        self.product= pd.DataFrame(produce_rate,columns=['product','rate','material','ratio'])
        self.product['total'] = [0.0] * len(produce_rate)
        self.product.set_index(['product'],inplace=True)
        # The dataframe of the container
        self.container = pd.DataFrame({'product':container, 'storage':[0.0]*len(container), 'capacity':[60000] * len(container)})
        self.container['source'] = self.container['product'].map(product_source)
        self.container.set_index(['product'],inplace=True)
        # self.container.at['P2','capacity'] = 2*capacity
        self.reset()

        # The number of pruduced component during last time step
        self.step_produced_num = 0
        self.step_final_product = 0
        # The number of decreased component during last time step
        self.step_transport = 0
        # self.step_emergency_product = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}
        # The penalty, when run out of material
        # self.penalty = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}

        # The number of trucks which desitination is current factory or stop at current factory
        self.truck_num = 0

        self.cumulate_reward = 0.0

        self.step = 0

    def reset(self) -> None:
        '''
        Set total storage and total producd to 0
        '''
        self.product['total'] = [0.0] * len(self.product)
        self.container['storage'] = [0.0]*len(self.container)
        self.cumulate_reward = 0.0

    
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

            # Update the reward for factory agent
            # self.step_produced_num += item_num

            tmp_materials = row['material']
            if type(tmp_materials) == str:
                tmp_materials = tmp_materials.split(',')
                tmp_ratio = np.array(row['ratio'].split(','),dtype=np.float64)

                tmp_storage = self.container.loc[tmp_materials,'storage'].to_numpy()

                # # Calculate Penalty. If the raw material, give a penalty
                # for remain_material, single_ratio in zip(tmp_materials,tmp_ratio):
                #     if self.container.loc[remain_material,'storage'] <= single_ratio*tmp_rate:
                #         tmp_source_factory = self.container.loc[remain_material,'source']
                #         self.penalty[tmp_source_factory] -= 0.05

                # Check storage
                if (tmp_storage > tmp_ratio*tmp_rate).all() and self.container.loc[index,'capacity'] > self.container.loc[index,'storage']:
                    # Consume the material
                    for i in range(len(tmp_materials)):
                        self.container.at[tmp_materials[i],'storage'] = self.container.loc[tmp_materials[i],'storage'] - item_num * tmp_ratio[i]
                    # Produce new product
                    self.container.at[index,'storage'] = self.container.loc[index,'storage'] + item_num
                    self.product.at[index,'total'] = self.product.loc[index,'total'] + item_num

                    # Only record the product which need raw materials
                    self.step_produced_num += item_num
                    if index == 'A' or index == 'B':
                        self.step_final_product += item_num

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
            # Maximum loading speed: 0.5 t/s
            load_weight = min(0.5, self.container.loc[product,'storage'])
            truck_state, exceed_cargo =  truck.load_cargo(weight=load_weight, product= product)
            self.container.at[product,'storage'] = self.container.loc[product,'storage'] - (load_weight - exceed_cargo)
            self.step_transport += load_weight - exceed_cargo
            return truck_state
    
    def unload_cargo(self, truck:Truck) -> None:
        '''
        Unload cargo to container
        '''
        if self.id in truck.position and (truck.state == 'pending for unloading' or truck.state == 'unloading') and self.container.loc[truck.product,'storage'] < self.container.loc[truck.product,'capacity']:
            unload_weight = min(0.5, self.container.loc[truck.product,'capacity'] - self.container.loc[truck.product,'storage'])
            truck_state, exceed_cargo = truck.unload_cargo(unload_weight)
            self.container.at[truck.product,'storage'] = self.container.loc[truck.product,'storage'] + (unload_weight - exceed_cargo)
            # product_source = self.container.loc[truck.product,'source']
            # self.step_emergency_product[product_source] += emergency_par * (unload_weight - exceed_cargo)
        
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
            '''
            Start loading the product to truck.
            Only when the product is enough to full the truck
            '''
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

    def base_produce_load(self) -> None:
        for tmp_factory in self.factory:
            # Produce the product
            tmp_factory.produce_product()
            # Unload goods for all truck
            for tmp_truck in self.truck:
                tmp_factory.unload_cargo(tmp_truck)
            '''
            Start loading the product to truck.
            Only when the product is enough to full the truck
            '''
            # Get the index of the transported product
            tmp_product = self.product_idx[tmp_factory.id]
            
            # Get the list of trucks that start loading goods
            truck_continue = [truck for truck in self.truck if truck.position == tmp_factory.id and truck.state == 'loading']
            for tmp_truck in truck_continue:
                if tmp_truck.position == tmp_factory.id:
                    tmp_result = tmp_factory.load_cargo(tmp_truck,tmp_truck.product)
                    if tmp_result == 'full':
                        # print(f'[delievery] {tmp_truck.id} delivers the {tmp_truck.product}')
                        tmp_truck.delivery(self.transport_idx[tmp_truck.product])
                        # Remove the left truck from pool
                        truck_continue.remove(tmp_truck)
            
            # Get the list of trucks that wait to load goods
            truck_pool = [truck for truck in self.truck if truck.position == tmp_factory.id and truck.state == 'waitting']
            # Check the current storage, decide whether loading the goods
            # Random select the goods to transport
            random.shuffle(tmp_product)
            for loading_product in tmp_product:
                # Refresh the trucks' number
                num_truck = len([truck for truck in truck_continue if truck.product == loading_product])
                # Iterate the truck pool
                for tmp_truck in truck_pool:
                    # Check the storage of the loading_product
                    # If it's enouth for current activate truck, assgin new trucks
                    if (num_truck+1) * tmp_truck.capacity <= tmp_factory.container.loc[loading_product,'storage']:
                        tmp_result = tmp_factory.load_cargo(tmp_truck,loading_product)
                        # Refresh the activate trucks' number
                        num_truck += 1
                    else:
                        break

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
    
    def truck_agents(self):
        return [agent for agent in self.agents if agent.truck]
    
    def factory_agents(self):
        return [agent for agent in self.agents if not agent.truck]
    
    def resume_truck(self):
        '''
        resume all truck from parking area to get the distance
        '''
        for agent in self.agents:
            if agent.truck:
                tmp_pk = traci.vehicle.getStops(vehID=agent.id)
                if len(tmp_pk) > 0:
                    latest_pk = tmp_pk[0]
                    if latest_pk.arrival > 0:
                        traci.vehicle.resume(vehID=agent.id)
                # try:
                #     traci.vehicle.resume(vehID=agent.id)
                # except:
                #     pass
        
        traci.simulationStep()
    
    def park_truck(self):
        '''
        put all truck back to the parking area
        '''
        for agent in self.agents:
            if agent.truck:
                try:
                    traci.vehicle.setParkingAreaStop(vehID=agent.id, stopID=agent.destination)
                except:
                    pass
        for _ in range(5):
            traci.simulationStep()
    
    def flag_reset(self):
        for factory_agent in self.manager.factory:
            # The number of pruduced component during last time step
            factory_agent.step_produced_num = 0
            factory_agent.step_final_product = 0
            # The number of decreased component during last time step
            factory_agent.step_transport = 0
            # factory_agent.step_emergency_product = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}
            # The penalty, when run out of material
            # factory_agent.penalty = {'Factory0':0, 'Factory1':0, 'Factory2':0, 'Factory3':0}