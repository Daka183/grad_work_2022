import pandas as pd
from os.path import join

class Loader():

    def load_data_csv(self, mode):
        path = 'C:/Users/aslop/Downloads/data_in_csv/'

        if mode != 'check_real_data':
            return (pd.read_csv(join(path, 'train_40k.csv')), 
                pd.read_csv(join(path, 'test_10k.csv')))   
        else:  
            return pd.read_csv(join(path, 'real_data.csv'))

