import traci
import numpy as np
import pandas as pd
import random

class product_management(object):
    '''
    This class is based on the industry 4.0 project
    There are 2 final products: A(P123) and B(P234)
    '''
    
    def __init__(self, factory:list, lorry:list) -> None:
        '''
        Input the list of factories and the lorries
        Producding order:
        Factory0: produce P1
        Facotry1: produce P2, P12
        Factory2: produce P3, P23, A(P123)
        Factory3: produce P4, B(P234)
        '''
        self.factory = factory
        self.lorry = lorry
        self.p = np.array([1.0, 1.0, 1.0, 1.0, lorry[0].capacity])
        self.et = 100
        self.s = 0
        self.s1 = 0
        self.s2 =0
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
            for tmp_lorry in self.lorry:
                tmp_factory.unload_cargo(tmp_lorry)
            # Start loading the product to lorry.
            # Only when the product is enough to full the lorry
            tmp_product = self.product_idx[tmp_factory.id]
            lorry_pool = [lorry for lorry in self.lorry if lorry.position == tmp_factory.id and lorry.state == 'waitting']

            # Continue loading
            lorry_continue = [lorry for lorry in self.lorry if lorry.position == tmp_factory.id and lorry.state == 'loading']
            for tmp_lorry in lorry_continue:
                if tmp_lorry.position == tmp_factory.id:
                    tmp_result = tmp_factory.load_cargo(tmp_lorry,tmp_lorry.product)
                    if tmp_result == 'full':
                        # print(f'[delievery] {tmp_lorry.id} delivers the {tmp_lorry.product}')
                        tmp_lorry.delivery(self.transport_idx[tmp_lorry.product])
            
            # for item in tmp_product:
                # print(item not in lorry_duplicate)
            lorry_duplicate = [lorry.product for lorry in self.lorry if lorry.position == tmp_factory.id and lorry.state == 'loading']
            if len(tmp_product) == 2:
                # loading the product with max storage
                item = self.factory[2].container.loc[tmp_product,'storage'].idxmin()
                item_bak = [i for i in tmp_product if i != item][0]
                if (tmp_factory.container.loc[item,'storage'] > self.lorry[0].capacity) and (item not in lorry_duplicate) and (len(lorry_pool)>0):
                    tmp_result = tmp_factory.load_cargo(lorry_pool[0],item)
                elif (tmp_factory.container.loc[item_bak,'storage'] > self.lorry[0].capacity) and (item_bak not in lorry_duplicate) and (len(lorry_pool)>0):
                    tmp_result = tmp_factory.load_cargo(lorry_pool[0],item_bak)

            elif len(tmp_product) == 1:
                item = tmp_product[0]
                if (tmp_factory.container.loc[item,'storage'] > self.lorry[0].capacity) and (item not in lorry_duplicate) and (len(lorry_pool)>0):
                    tmp_result = tmp_factory.load_cargo(lorry_pool[0],item)

    def lorry_manage(self) -> None:
        s1 = np.zeros(len(self.factory))
        # Only use normal lorry
        lorry_count = np.array([i.position for i in self.lorry if i.state != 'broken' and i.state != 'repair' and i.state != 'maintenance'])
        if len(lorry_count) == 0:
            lorry_count = np.array(['all broken'])
        n_lorry = {'Factory0':np.count_nonzero(lorry_count=='Factory0'),
                  'Factory1':np.count_nonzero(lorry_count=='Factory1'),
                  'Factory2':np.count_nonzero(lorry_count=='Factory2'),
                  'Factory3':np.count_nonzero(lorry_count=='Factory3')}
        
        lorry_p1 = np.sum([i.weight for i in self.lorry if i.product == 'P1' and i.state == 'delivery'])
        lorry_p2 = np.sum([i.weight for i in self.lorry if i.product == 'P2' and i.state == 'delivery'])
        lorry_p12 = np.sum([i.weight for i in self.lorry if i.product == 'P12' and i.state == 'delivery'])
        lorry_p23 = np.sum([i.weight for i in self.lorry if i.product == 'P23' and i.state == 'delivery'])

        mA = min(self.factory[0].container.loc['P1','storage'], int(len(self.lorry)/2)*self.lorry[0].capacity) - (self.factory[1].container.loc['P1','storage'] + lorry_p1)
        s1[0] = self.p[0] * mA
        mB1 = min(self.factory[1].container.loc['P2','storage'] - self.factory[1].container.loc['P1','storage'], int(len(self.lorry)/2)*self.lorry[0].capacity) - (self.factory[2].container.loc['P2','storage'] + lorry_p2)
        mB2 = min(self.factory[1].container.loc['P12','storage'], int(len(self.lorry)/2)*self.lorry[0].capacity) - (self.factory[2].container.loc['P12','storage'] + lorry_p12)
        s1[1] = self.p[1] * mB1 + self.p[2] * mB2
        mC = min(self.factory[2].container.loc['P23','storage'], int(len(self.lorry)/2)*self.lorry[0].capacity) - (self.factory[3].container.loc['P23','storage'] + lorry_p23)
        s1[2] = self.p[3] * mC
        s2 = - self.p[4] * np.array(list(n_lorry.values()))
        s2_pool = - self.p[4] * (np.array(list(n_lorry.values())) - 1)
        s = s1 + s2

        # Generate the lorry pool
        s_pool = (s1 + s2_pool<=0)
        # lorry in the factory_idx could be assigned
        factory_idx = [self.factory[i].id for i in np.where(s_pool==True)[0]]
        lorry_pool = [i for i in self.lorry if i.position in factory_idx and i.state == 'waitting']

        # Assign the lorry
        self.s = s
        self.s1 = s1
        self.s2 = s2
        if np.max(s) > 0 and len(lorry_pool) >0 :
            factory_assign = self.factory[np.argmax(s)].id
            c = np.zeros(len(lorry_pool))
            for i in range(len(lorry_pool)):
                tmp_lorry = lorry_pool[i]
                if tmp_lorry.position == 'Factory3':
                    c[i] = -1
                    break
                else:
                    tmp_des = tmp_lorry.destination
                    traci.vehicle.changeTarget(vehID=tmp_lorry.id,edgeID=factory_assign)
                    c[i] = traci.vehicle.getDrivingDistance(vehID=tmp_lorry.id, edgeID=factory_assign,pos=0)
                    traci.vehicle.changeTarget(vehID=tmp_lorry.id,edgeID=tmp_des)
            
            #print(f'[Assign] {lorry_pool[c.argmin()].id} relocates to {factory_assign}')
            lorry_pool[c.argmin()].delivery(destination=factory_assign)
    
    def base_line(self) -> None:
        # Create a lorry pool for all factory
        lorry_pool = {'Factory0': [tmp_lorry for tmp_lorry in self.lorry if tmp_lorry.position == 'Factory0' and tmp_lorry.state == 'waitting'],
                      'Factory1': [tmp_lorry for tmp_lorry in self.lorry if tmp_lorry.position == 'Factory1' and tmp_lorry.state == 'waitting'],
                      'Factory2': [tmp_lorry for tmp_lorry in self.lorry if tmp_lorry.position == 'Factory2' and tmp_lorry.state == 'waitting'],
                      'Factory3': [tmp_lorry for tmp_lorry in self.lorry if tmp_lorry.position == 'Factory3' and tmp_lorry.state == 'waitting']}
        #Free lorry in Factory 2 and 3 go back to Factory 0 or 1
        ass_des = ['Factory0','Factory1']
        for tmp_lorry in lorry_pool['Factory2']:
            tmp_des = ass_des[random.randint(0,1)]
            tmp_lorry.delivery(destination=tmp_des)
        for tmp_lorry in lorry_pool['Factory3']:
            tmp_des = ass_des[random.randint(0,1)]
            tmp_lorry.delivery(destination=tmp_des)
        