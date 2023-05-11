import traci
import numpy as np
import pandas as pd

from .lorry import Lorry

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

        # Create a dataframe to record the products which are produced in current factory
        self.product= pd.DataFrame(produce_rate,columns=['product','rate','material','ratio'])
        self.product['total'] = [0.0] * len(produce_rate)
        self.product.set_index(['product'],inplace=True)
        # The dataframe of the container
        self.container = pd.DataFrame({'product':container, 'storage':[0.0]*len(container), 'capacity':[capacity] * 4 + [60000] * 4})
        self.container.set_index(['product'],inplace=True)
        self.container.at['P2','capacity'] = 2*capacity

        self.step = 0
    
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



