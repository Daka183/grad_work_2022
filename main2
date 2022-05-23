from data_helpers import main_data_loader
from nn_models import create_cnn_model, save_model, restore_model
from visualization import quality_metrics_show
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import pandas as pd
import os
import json
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping
import argparse
import os
from keras.callbacks import LearningRateScheduler

import warnings
warnings.filterwarnings('ignore')

args = None
data = {}


class Metrics(Callback):

    def __init__(self,validation_data):
        self.val_data = validation_data

    # Metrics for epochs
    def main_metrics(self):
        init_data = self.init_data
        X_test = init_data[0]
        y_test = init_data[1]
        y_pred = self.model.predict(X_test, batch_size = args.batch_size)
        y_pred = np.argmax(y_pred, axis = 1)
        y_test = np.argmax(y_test, axis = 1)
        return [accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='micro'),
          recall_score(y_test, y_pred, average='micro')]


    def epoch_print(self, epoch, logs={}):
        acc, precision, recall = self.main_metrics()
        print('\nFor epoch %d: Acc: %0.4f, Prec: %0.2f, Rec: %0.2f'%(epoch, acc, precision, recall))
        print((str(acc) + '; ' +  str(precision) + '; ' + str(recall)).replace('.', ','))


def quality_metrics_calc(y_test, y_pred, level):
    accuracy = accuracy_score(y_test, y_pred)
    precision =  precision_score(y_test, y_pred, average='macro').round(3)
    recall = recall_score(y_test, y_pred, average='macro').round(3)  
    results = [accuracy, precision, recall]

    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))

    quality_metrics_show(y_test, y_pred, level)

    return results

def train(model, early_stop = True):

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../models', args.filename + '.json')
    if os.path.exists(model_path):
        print('Loading existed model')

    else:
        print('Training Model')
        callbacks_list = []
        lr_decay = LearningRateScheduler(schedule=lambda epoch: args.lr_rate)
        callbacks_list.append(lr_decay)
        if early_stop:
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto')
            callbacks_list.append(early_stop)
        else:
            metrics_callback = Metrics(validation_data = (data['X_test'], data['y_test']))
            callbacks_list.append(metrics_callback)
            model.fit(data['X_train'], data['y_train'], batch_size=args.batch_size, epochs=args.epochs, verbose=1,
            callbacks = callbacks_list)

        save_model(model, filename=args.filename)

def test(model, X_test, y_test, y_test_1, y_test_2):

    global data
    print('Testing Model...')

    y_pred = model.predict(X_test, batch_size = args.batch_size)
    y_pred = pd.Series(np.argmax(y_pred, axis = 1), name='join_column') 
    y_test = pd.Series(np.argmax(y_test, axis = 1), name='join_column') 

    y_test_name = pd.read_csv(os.path.join(os.path.dirname(__file__), '../resources', 'y_test.csv'), names=['classes'])
    y_test_concat = pd.concat([y_test, y_test_name], axis=1)[['join_column', 'classes']]
    y_test_concat = y_test_concat.drop_duplicates()
    results = y_test_concat.merge(y_pred, on='join_column', how='right')
    split_data = results['classes'].str.split('/', expand=True)
    y_test_concat.to_csv(os.path.join(os.path.dirname(__file__), '../resources', 'y_test_concat.csv'))

    y_pred_1 = split_data[0]
    y_pred_2 = split_data[1]

    print('\nTotal metrics:')
    res_1 = quality_metrics_calc(y_test, y_pred, 0)
    print('\nMetrics for Executor:')
    res_2 = quality_metrics_calc(y_test_1, y_pred_1, 1)
    print('\nMetrics for Department:')
    res_3 = quality_metrics_calc(y_test_2, y_pred_2, 2)

    return [res_1, res_2, res_3]


def model_cnn():
    return create_cnn_model(args.embed_dim, args.seq_len,
        args.num_filters,len(data['y_train'][0]),
        data['vocabulary'], args.lr_rate, args.lang)


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', nargs='+', default=None, help='Input message')
    parser.add_argument('--lang', type=str, default='russian', choices=['russian','english'], help='Language')
    parser.add_argument('--mode', type=str, default='train_test', choices=['research', 'train_test', 'check_real_data'], help='Mode of the program')
    parser.add_argument('--data_type', type=str, default='csv', choices=['csv','cmd'], help='Data type')
    parser.add_argument('--batch_size', type=int, default=32, help = 'Batch size')
    parser.add_argument('--seq_len', type=int, default=200, help = 'Sequence length')
    parser.add_argument('--epochs', type=int, default=1, help = 'Number of epochs')
    parser.add_argument('--num_filters', type=int, default=512, help = 'Number of filters')
    parser.add_argument('--embed_dim', type=int, default=300, help = 'Embedding dimension')
    parser.add_argument('--lr_rate', type = float, default = 0.001, help = 'Learning rate')
    parser.add_argument('--early_stop', action='store_true', default = False, help = 'Early stopping')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent = 2))
    run()


def run():

    if args.mode == 'check_real_data' or args.data_type == 'cmd':
        X_real = init_real_data()
        model = restore_model()

        print('Get results \n')
        pred = model.predict(X_real, batch_size = args.batch_size)
        pred = pd.Series(np.argmax(pred, axis = 1), name='join_column') 
        y_test_concat = pd.read_csv(os.path.join(os.path.dirname(__file__), '../resources', 'y_test_concat.csv'))
        results = y_test_concat.merge(pred, on='join_column', how='right')
        split_data = results['classes'].str.split('/', expand=True)

        if args.data_type == 'csv':
            messages = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets', 'real_data.csv'))
            messages['class_1'] = split_data[0]
            messages['class_2'] = split_data[1]

            mess_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
            if not os.path.exists(mess_dir_save_path):
                os.makedirs(mess_dir_save_path)

            writer = pd.ExcelWriter(os.path.join(os.path.dirname(__file__), '../results', 'results.xlsx'), engine='xlsxwriter')
            pd.DataFrame(messages).to_excel(writer, sheet_name='results')
            writer.save()

            print('Results saved')

        elif args.data_type == 'cmd':
            message = ' '.join(args.text)
            executer = split_data.iloc[0,0]
            department = split_data.iloc[0,1]

            print('Your message:\n', message)
            print('Executer:', executer)
            print('Department:', department)

    elif args.mode =='train_test':
        init_data()
        model = create_model()
        train(model,  early_stop = args.early_stop)
        test(model, X_test = data['X_test'], y_test = data['y_test'], 
            y_test_1=data['y_test_1'], y_test_2=data['y_test_2'])

        print(args.filename)

    elif args.mode == 'research':

        params = ['relu']
        results = np.zeros((3,3))

        research_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research')
        if not os.path.exists(research_dir_save_path):
            os.makedirs(research_dir_save_path)

        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../research', 'test.txt')
        out_file = open(filename, 'w')
        metrics = ['Accuracy', 'Precision', 'Recall']

        init_data()
        for param in params:
            # args.threshold = param
            model = create_model(param)
            train(model, early_stop = False)
            results[0], results[1], results[2] = test(model, X_test = data['X_test'], y_test = data['y_test'],
            y_test_1=data['y_test_1'], y_test_2=data['y_test_2'])

            out_file.write('_________Results with:' + str(param) + '_________\n')
            for i in range(results.shape[0]):
                out_file.write('\nCat'+str(i)+':\n')
                for j in range(results.shape[1]):
                    out_file.write(metrics[j] + ': ' + str(results[i][j]))
                    out_file.write('\n')
            out_file.write('\n')
        out_file.close()

    K.clear_session()

# Create filename and reload or create model
def create_model():
    args.filename = ('cnnF-' + str(args.num_filters) + '_dT-' + str(args.data_type) + '_bS-' 
    + str(args.batch_size) + '_E-' + str(args.epochs) + '_sL-' + str(args.seq_len) 
    + '_lR-' + str(args.lr_rate) + '_L-' + str(args.lang))

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '../models', args.filename + '.json')
    if os.path.exists(model_path):
        model = restore_model(args.filename)
        return model
    else:
        return model_cnn()

# Data initialiation
def init_data():
    global data
    X_train, y_train, X_test, y_test, vocabulary, y_test_1, y_test_2 = main_data_loader(seq_length_max =  args.seq_len, 
        data_type = args.data_type, lang = args.lang)

    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_test'] = X_test
    data['y_test'] = y_test
    data['vocabulary'] = vocabulary
    data['y_test_1'] = y_test_1
    data['y_test_2'] = y_test_2


def init_real_data():

    if args.data_type == 'cmd':
        text_cmd = ' '.join(args.text)
        text_cmd = pd.Series(text_cmd, name='Letter')
    else:
        text_cmd = ''

    X = main_data_loader(X = text_cmd, seq_length_max =  args.seq_len, 
        data_type = args.data_type, mode = args.mode, lang = args.lang)
    return X


if __name__ == '__main__':
    main()
