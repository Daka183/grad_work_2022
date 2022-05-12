from data_helpers import main_data_loader
from nn_models import create_cnn_model, create_lstm_model, save_model, restore_model
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
import scipy
from keras.callbacks import LearningRateScheduler

import warnings
warnings.filterwarnings('ignore')

args = None
data = {}


class Metrics(Callback):

    def __init__(self,validation_data):
        self.val_data = validation_data

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
        print("\nFor epoch %d: Acc: %0.4f, Prec: %0.2f, Rec: %0.2f"%(epoch, acc, precision, recall))
        print((str(acc) + '; ' +  str(precision) + '; ' + str(recall)).replace(".", ","))


def quality_metrics_calc(y_test, y_pred, level):
    accuracy = accuracy_score(y_test, y_pred)
    precision =  precision_score(y_test, y_pred, average='macro').round(3)
    recall = recall_score(y_test, y_pred, average='macro').round(3)  
    results = [accuracy, precision, recall]

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

    quality_metrics_show(y_test, y_pred, level)

    return results

def train(model, early_stop = True):

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../models', args.filename + '.json')
    if os.path.exists(model_path):
        print("Loading existed model")

    else:
        print("Training Model")
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
    print("Testing Model...")

    y_pred = model.predict(X_test, batch_size = args.batch_size)
    y_pred = np.argmax(y_pred, axis = 1)
    y_test = np.argmax(y_test, axis = 1)

    # pred = pd.DataFrame(y_pred).idxmax(axis=1)
    pred = pd.Series(y_pred, name='join_column')
    labels = pd.read_csv(os.path.join(os.path.dirname(__file__), '../resources', 'labels_train.csv'))
    results = labels.merge(pred, on='join_column', how='right')
    split_data = results['classes'].str.split('/', expand=True)

    y_pred_1 = split_data[0]
    y_pred_2 = split_data[1]

    print('\nTotal metrics:')
    res_1 = quality_metrics_calc(y_test, y_pred, 0)
    print('\nMetrics for Cat1:')
    res_2 = quality_metrics_calc(y_test_1, y_pred_1, 1)
    print('\nMetrics for Cat2:')
    res_3 = quality_metrics_calc(y_test_2, y_pred_2, 2)

    # writer = pd.ExcelWriter('c:/users/aslop/Downloads/df_test.xlsx', engine='xlsxwriter')
    # pd.DataFrame(y_test).to_excel(writer, sheet_name='y_test')
    # pd.DataFrame(y_pred).to_excel(writer, sheet_name='y_pred')
    # pd.DataFrame(y_test_1).to_excel(writer, sheet_name='y_test_1')
    # pd.DataFrame(y_pred_1).to_excel(writer, sheet_name='y_pred_1')
    # pd.DataFrame(y_test_2).to_excel(writer, sheet_name='y_test_2')
    # pd.DataFrame(y_pred_2).to_excel(writer, sheet_name='y_pred_2')
    # writer.save()

    return [res_1, res_2, res_3]


def model_cnn(param=None):
    return create_cnn_model(args.embed_dim, args.seq_len,
        args.num_filters,len(data['y_train'][0]),
        data['vocabulary'], args.lr_rate, param)


def model_lstm():
    return create_lstm_model(args.embed_dim, args.seq_len,
        args.lstm_units, len(data['y_train'][0]),
        data['vocabulary'], args.lr_rate)


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_test', choices=['research', 'train_test', 'check_real_data'], help="Mode of the program")
    parser.add_argument('--data_type', type=str, default='csv', choices=['csv','cmd'], help="Data type")
    parser.add_argument('--network', type=str, default='cnn', choices=['cnn','lstm'], help="Classifier architecture")
    parser.add_argument('--batch_size', type=int, default=32, help = 'Batch size')
    parser.add_argument('--seq_len', type=int, default=200, help = "Sequence length")
    parser.add_argument('--epochs', type=int, default=1, help = "Number of epochs")
    parser.add_argument('--lstm_units', type=int, default=700, help = "Units in LSTM")
    parser.add_argument('--num_filters', type=int, default=512, help = "Filters in CNN")
    parser.add_argument('--embed_dim', type=int, default=300, help = "Embedding dim size")
    parser.add_argument('--early_stop', action='store_true', default = False, help = 'Early stopping')
    parser.add_argument('--lr_rate', type = float, default = 0.001, help = 'Learning rate')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent = 2))
    run()


def run():

    if args.mode =='train_test':
        init_data()
        model = create_model()
        train(model,  early_stop = args.early_stop)
        test(model, X_test = data['X_test'], y_test = data['y_test'], 
            y_test_1=data['y_test_1'], y_test_2=data['y_test_2'])

        print(args.filename)

    elif args.mode == 'research':

        params = ['relu', 'sigmoid', 'softmax', 'softplus', 'tanh', 'selu', 'elu', 'exponential']
        results = np.zeros((3,3))

        research_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research')
        if not os.path.exists(research_dir_save_path):
            os.makedirs(research_dir_save_path)

        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../research', 'test_act_f.txt')
        out_file = open(filename, 'w')
        metrics = ['Accuracy', 'Precision', 'Recall']

        init_data()
        for param in params:
            # args.threshold = param
            model = create_model(param)
            train(model, early_stop = False)
            results[0], results[1], results[2] = test(model, X_test = data['X_test'], y_test = data['y_test'],
            y_test_1=data['y_test_1'], y_test_2=data['y_test_2'])

            out_file.write("_________Results with:" + str(param) + '_________\n')
            for i in range(results.shape[0]):
                out_file.write('\nCat'+str(i)+':\n')
                for j in range(results.shape[1]):
                    out_file.write(metrics[j] + ": " + str(results[i][j]))
                    out_file.write('\n')
            out_file.write('\n')
        out_file.close()

    elif args.mode =='check_real_data':
        X_real = init_real_data()
        model = restore_model()

        print("Get results")
        pred = model.predict(X_real, batch_size = args.batch_size)
        # pred = np.argmax(pred, axis = 1)
        
        pred = pd.DataFrame(pred).idxmax(axis=1)
        pred = pd.Series(pred, name='join_column')
        labels = pd.read_csv(os.path.join(os.path.dirname(__file__), '../resources', 'labels_train.csv'))
        results = labels.merge(pred, on='join_column', how='right')
        split_data = results['classes'].str.split('/', expand=True)

        messages = pd.read_csv('C:/Users/aslop/Downloads/data_in_csv/real_data.csv')
        messages['class_1'] = split_data[0]
        messages['class_2'] = split_data[1]

        mess_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
        if not os.path.exists(mess_dir_save_path):
            os.makedirs(mess_dir_save_path)

        writer = pd.ExcelWriter(os.path.join(os.path.dirname(__file__), '../results', 'results.xlsx'), engine='xlsxwriter')
        pd.DataFrame(messages).to_excel(writer, sheet_name='results')
        writer.save()

        print("Results saved")

    K.clear_session()


def create_model(param=None):
    general_name = ("_dT-" + str(args.data_type)+ "_bS-" + str(args.batch_size) + 
    "_E-" + str(args.epochs) + "_sL-" + str(args.seq_len) + "_lR-" + str(args.lr_rate))
    if args.network == 'lstm':
        args.filename = ('lstmU-' + str(args.lstm_units) + general_name)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../models', args.filename + '.json')
        if os.path.exists(model_path):
            model = restore_model(args.filename)
            return model
        else:
            return model_lstm()
    elif args.network == 'cnn':
        args.filename = ('cnnF-' + str(args.num_filters) + general_name)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../models', args.filename + '.json')
        if os.path.exists(model_path):
            model = restore_model(args.filename)
            return model
        else:
            return model_cnn(param)


def init_data():
    global data
    X_train, y_train, X_test, y_test, vocabulary, y_test_1, y_test_2 = main_data_loader(seq_length_max =  args.seq_len, 
        data_type = args.data_type)

    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_test'] = X_test
    data['y_test'] = y_test
    data['vocabulary'] = vocabulary
    data['y_test_1'] = y_test_1
    data['y_test_2'] = y_test_2

def init_real_data():
    X = main_data_loader(seq_length_max =  args.seq_len, 
        data_type = args.data_type, mode = args.mode)
    return X


if __name__ == '__main__':
    main()
