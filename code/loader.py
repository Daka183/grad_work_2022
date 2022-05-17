import pandas as pd
from os.path import join
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class Loader():

    def load_data_csv(self, mode, lang):
            
        path = 'C:/Program Files/Python39/Letters/'
        if lang == 'russian':
            df = pd.read_csv(join(path, 'letter_dataset.csv'))
        df = shuffle(df, random_state = 42)
        train, test = train_test_split(df, test_size=0.2)
        real_data = df.iloc[:20, :]

        if mode != 'check_real_data':
            return [train, test]
        else:
            real_data

