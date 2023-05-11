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
                 state:str = 'delivery', position:str = 'Factory0', destination:str = 'Factory0', product:str = 'A', eng=None, mdl:str=None,\
                 path:str = 'result', time_broken:int = 86400, mdp_freq:float = 6*3600, env_step:int = 3600, broken_freq:float = 86400*20,\
                 maintenance_freq:float = 4*3600, repair_freq:float = 86400*3) -> None:
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

        MDP parameter, unit is second
        time_broken: time based repair speed
        mdp_freq: mdp transfor frequency when lorry is running
        env_step: RL time step
        broken_freq: random failure frequency
        maintenance_freq: maintenance successful frequency
        repair_freq: repair succseeful frequency

        '''
        # Create lorry in sumo. If the lorry already exist, remove it first
        try:
            traci.vehicle.add(vehID=lorry_id,routeID=position + '_to_'+ destination, typeID='lorry')
        except:
            traci.vehicle.remove(vehID=lorry_id)
            traci.vehicle.add(vehID=lorry_id,routeID=position + '_to_'+ destination, typeID='lorry')
        traci.vehicle.setParkingAreaStop(vehID=lorry_id,stopID=position)
        
        self.id = lorry_id
        self.capacity = capacity
        self.weight = weight

        self.state = state
        self.position = position
        self.destination = destination
        self.product = product
        self.color = (255,255,0,255)
        self.recover_state = 'waitting'

        # sumo time
        self.time_step = 0
        # record total transported product
        self.total_product = 0.0

        # self.reward = pd.DataFrame({'time':[0.0], 'total_product':[0.0]})
        # self.reward.set_index(['time'], inplace=True)

        # Markov state
        #    5 (lambda_0 = 0.013364)
        #0 1 2 3 4 (lambda_1=0.333442, lambda_m=0.653194)
        self.mk_state = 0
        lm_0 = env_step/broken_freq
        lm_1 = env_step/mdp_freq
        self.mu0 = env_step/maintenance_freq
        self.mu1 = env_step/repair_freq
        self.threshold1 = lm_0
        self.threshold2 = 1-lm_1
        # Transfer the state after time 'state_trans'
        self.state_trans = env_step
        self.step = 1
        self.maintenance_step = 0
        self.broken_step = 0

        # recover after time_broken
        self.time_broken = time_broken # 1 day
        self.time_repair = 3600 * 4 # 4 hours
        self.frequency = 86400 # 1 day

        # Temporal repaired time step
        self.step_repair = 0
        self.path = path + '/lorry_record.csv'

        # matlab engine
        self.eng = eng
        self.mdl = mdl

        # maintenance flag, when true, the engine start maintenance
        # default: False
        self.maintenance_flag = False
        # episode flag
        self.episode_flag = False

        # sensor reading
        # self.sensor_store = pd.read_csv('sensor.csv',index_col=0)
        self.sensor_store = pd.DataFrame(
                             {'s1':[-0.261865793780688, -0.261865793780687, 0.854862119013063,  0,                  0.1, 0.3, 0.5, 0.7, 0,                  0],
                              's2':[0,                  -0.261865793780687, 0.854862119013063,  0,                  0.1, 0.3, 0.5, 0.7, 0,                  0],
                              's3':[0.232220609580154,  0.510258310975582,  0.854862119013064,  0,                  0.1, 0.3, 0.5, 0.7, 0,                  0],
                              's4':[0.41230769230978,   0.412307692307698,  0.854862119013063,  0,                  0.1, 0.3, 0.5, 0.7, 0,                  0],
                              's5':[0.615384615388344,  0.615384615385666,  0.854862119013063,  0,                  0.1, 0.3, 0.5, 0.7, 0,                  0],
                              's6':[0.854862119017747,  0.854862119015461,  0.854862119013063,  1.4255319149081,    0.1, 0.3, 0.5, 0.7, 1.4255319149081,    0],
                              's7':[1.0000000000064,    1.00000000000655,   1.00000000000091,   1.0000000000086,    0.1, 0.3, 0.5, 0.7, 0.412307692308142,  0],
                              's8':[1.16366612112214,   1.16366612112405,   0.854862119012898,  1.42553191490811,   0.1, 0.3, 0.5, 0.7, 0,                  0],
                              's9':[1.42553191490218,   1.16366612112405,   0.854862119012898,  1.42553191490811,   0.1, 0.3, 0.5, 0.7, 0,                  0],
                              'MDPstate':[0,            1,                  2,                  3,                  4,   5,   6,   7,   8,                  9]}
                              )
        self.sensor = self.sensor_store.loc[self.sensor_store['MDPstate']==self.mk_state]

    
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
        self.time_step = time_step

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

        # fix some bug, ensure lorry stop when maintenance
        if self.mk_state > 3 and self.recover_state=='delivery' and len(tmp_pk)==1:
            self.lorry_stop()

        # Lorry maintenance
        if self.maintenance_flag:
            self.maintenance()
        elif self.state == 'maintenance':
            self.maintenance_step += 1
            if self.maintenance_step%self.state_trans==1:
                self.maintenance()
        
        # Repair the engine
        if repair_flag:
            self.repair()

        if self.state == 'broken':
            self.broken_step += 1
            if self.broken_step % self.state_trans == 1:
                self.broken_repair()
        # mannually repair the engine
        elif self.state == 'repair':
            self.step_repair +=1
            if self.step_repair % self.time_repair == 0:
                self.state = self.recover_state
                self.mk_state = 0
                self.step += 1
                # In sumo the lorry resume from stop
                # traci.vehicle.resume(vehID=self.id)
                self.lorry_resume()
                # print(f'[recover] {self.id}, mdp state: {self.mk_state}')
                with open(self.path,'a') as f:
                    f_csv = writer(f)
                    f_csv.writerow([self.time_step,self.id,self.mk_state,'recover after repaired'])
                
                # terminate the episode
                self.episode_flag = True
        # ignore the maintenance state
        elif self.mk_state > 3:
            pass
        elif parking_state.arrival < 0:
            self.state = 'delivery'
            if len(tmp_pk)>1:
                self.lorry_resume()
            self.step += 1
        elif self.weight == self.capacity and self.position == self.destination:
            self.state = 'pending for unloading'
        elif self.weight == 0:
            self.state = 'waitting'
        # Update the engine state and get sensor reading from Simulink
        if self.state == 'delivery' and self.step % self.state_trans ==0:
            self.MDP_model()
            if self.mk_state == 8 or self.mk_state == 9:
                # print(f'[Broken] {self.id}')
                self.state = 'broken'
                with open(self.path,'a') as f:
                    f_csv = writer(f)
                    f_csv.writerow([self.time_step,self.id,self.mk_state,'broken'])
                self.lorry_stop()
                
        self.sensor = self.sensor_store.loc[self.sensor_store['MDPstate']==self.mk_state]
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
    
    def repair(self):
        if ((self.time_step+1) % self.frequency == 0) and self.state != 'broken':
            self.mk_state = 0
            self.step = 1
            self.recover_state = self.state
            with open(self.path,'a') as f:
                f_csv = writer(f)
                f_csv.writerow([self.time_step,self.id,self.mk_state,'repairing'])
            # If the lorry is running, let it stop first
            if self.state == 'delivery':
                self.lorry_stop()
            
            self.state = 'repair'
            # print(f'[repair] {self.id} back to state {self.mk_state}')
    
    def broken_repair(self):
        lm = random.uniform(0,1)
        if lm < self.mu1:
            self.state = self.recover_state
            self.mk_state = 0
            self.step += 1
            # In sumo the lorry resume from stop
            if self.state == 'delivery':
                self.lorry_resume()
                # try:
                #     traci.vehicle.resume(vehID=self.id)
                # except:
                #     pass
            # print(f'[recover] {self.id}, mdp state: {self.mk_state}')
            self.broken_step = 0
            with open(self.path,'a') as f:
                f_csv = writer(f)
                f_csv.writerow([self.time_step,self.id,self.mk_state,'recover after broken'])
            
            # terminate the episode
            self.episode_flag = True
        else:
            pass
        
    def maintenance(self):
        self.maintenance_flag = False
        lm = random.uniform(0,1)
        if self.mk_state < 4:
            self.recover_state = self.state
            # If the lorry is running, let it stop first
            if self.state == 'delivery':
                self.lorry_stop()
            if lm < self.threshold1:
                self.mk_state = 9
                # print(f'[Broken] {self.id}')
                self.state = 'broken'
                # self.maintenance_flag = False
                return 'broken'
            else:
                self.mk_state += 4
                self.state = 'maintenance'
                # print(f'[maintenance] {self.id} go to state {self.mk_state}')
                with open(self.path,'a') as f:
                    f_csv = writer(f)
                    f_csv.writerow([self.time_step,self.id,self.mk_state,'maintenance'])
                return 'maintenance'
        elif self.mk_state >= 4 and self.mk_state < 8:
            if lm < self.mu0:
                self.mk_state = max(0, self.mk_state-5)
                self.state = self.recover_state
                # In sumo the lorry resume from stop
                if self.state == 'delivery':
                    self.lorry_resume()
                    # try:
                    #     traci.vehicle.resume(vehID=self.id)
                    # except:
                    #     pass
                # self.maintenance_flag = False
                # print(f'[recover] {self.id}, mdp state: {self.mk_state}')
                self.maintenance_step = 0
                with open(self.path,'a') as f:
                    f_csv = writer(f)
                    f_csv.writerow([self.time_step,self.id,self.mk_state,'recover after maintenance'])

                # terminate the episode
                self.episode_flag = True
                return 'successful'
            else:
                return 'try again'
        else:
            return 'broken'

    
    def MDP_model(self) -> None:
        '''
        Update the Simulink model
        State 0: Normal
        State 1: Clutch A, locked fault
        State 2: Clutch B,C  locked fault
        State 3: Clutch D,E locked fault, F unlocked fault
        State 4: Clutch A,B unlocked fault, E,F locked fault
        State 5: Clutch A,B,C,D,E,F unlocked fault

        lm is a random number. 
        If lm < 0.013364, Markov state go to 5. 
        If lm > 0.666558, Markov state pluse 1
        Otherwise, no change
        '''
        # self.sensor.drop(self.sensor.index, inplace=True)
        lm = random.uniform(0,1)
        if lm < self.threshold1:
            self.mk_state = 9
        elif lm > self.threshold2:
            if self.mk_state < 3:
                self.mk_state += 1
            elif self.mk_state == 3:
                self.mk_state = 8
        else:
            self.mk_state = self.mk_state
        # print(f'[MDP state] {self.id} state: {self.mk_state}, time:{self.time_step}')
        
        