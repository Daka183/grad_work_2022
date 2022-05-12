from loader import Loader 
import numpy as np
import pandas as pd
import os
import itertools
from collections import Counter
from preprocessing import text_cleaning, lemmatization
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from visualization import word_cloud_show, distrubution_show, word_cloud_by_classes_show
import xlsxwriter
import pickle

mb = MultiLabelBinarizer()
loader = None


def get_process_data (dataset, data_type, mode):

    if data_type == 'csv' and mode !='check_real_data':

        train, test = loader.load_data_csv(mode)

        distrubution_show(train)

        X_train, y_train_1, y_train_2 = (train['Text'], train['Cat1'], train['Cat2'])
        X_test, y_test_1, y_test_2 = (test['Text'], test['Cat1'], test['Cat2'])

        y_train = y_train_1 + '/' + y_train_2
        y_test = y_test_1 + '/' + y_test_2

        X_train = preprocessing_data(X_train, data_type)
        X_test = preprocessing_data(X_test, data_type)
 
        #X_train, X_test = vectorization (X_train, X_test)

        #word_cloud_by_classes_show(X_train, y_train)

        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
        y_test = y_test.reindex(columns = y_train.columns, fill_value=0)
        # if labels.count(axis=0) != np.count_nonzero(y_test.count(axis=1):
        #     print ('Number of classes in test df is not equal number of classes in train df!')

        return [X_train, X_test, y_train, y_test, y_test_1, y_test_2]
    
    elif data_type == 'csv' and mode =='check_real_data':

        X = loader.load_data_csv(mode)

        X = X['Text']

        X = preprocessing_data(X, data_type)
 
        #X_train, X_test = vectorization (X_train, X_test)

        return X        


def preprocessing_data(x_text, data_type):

    print("Preprocessing start")

    x_text = text_cleaning(x_text)
    x_text = lemmatization (x_text)

    return x_text


def create_vocabulary(sentences):
    word_count = Counter(itertools.chain(*sentences))
    vocab_inv = [word[0] for word in word_count.most_common()]
    vocab_inv = list(sorted(vocab_inv))
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    return vocab


def np_transrofm(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def main_data_loader(seq_length_max = 200, data_type = 'csv', mode = 'train_test'):
    global loader
    loader = Loader()

    if data_type == 'csv' and mode != 'check_real_data':
        X_train, X_test, y_train, y_test, y_test_1, y_test_2 = get_process_data(type, data_type, mode)

        sentence_pad_train = pad_sequences(X_train, maxlen=seq_length_max, dtype='str', padding = 'post', truncating ='post')
        sentence_pad_train = [sentence[:seq_length_max] + ['0'] * (seq_length_max - len(sentence)) for sentence in X_train]
        vocabulary_train = create_vocabulary(sentence_pad_train)
        x_train, y_train = np_transrofm(sentence_pad_train, y_train, vocabulary_train)

        X_test = [[a for a in sentence if a in vocabulary_train] for sentence in X_test]
        sentences_padded_test = pad_sequences(X_test, maxlen=seq_length_max, dtype='str', padding = 'post', truncating ='post')
        sentences_padded_test = [sentence[:seq_length_max] + ['0'] * (seq_length_max - len(sentence)) for sentence in X_test]
        x_test, y_test = np_transrofm(sentences_padded_test, y_test, vocabulary_train)

        word_cloud_show (vocabulary_train)

        vocab_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vocabularies')
        if not os.path.exists(vocab_dir_save_path):
            os.makedirs(vocab_dir_save_path)
        vocab_save_path = os.path.join(os.path.dirname(__file__), '../vocabularies', 'data_vocab.vc')
        with open(vocab_save_path, 'wb') as output:
            pickle.dump(vocabulary_train, output)

        return [x_train, y_train, x_test, y_test, vocabulary_train, y_test_1, y_test_2]

    elif data_type == 'csv' and mode =='check_real_data':
        X = get_process_data(type, data_type, mode)

        vocab_saved_path = os.path.join(os.path.dirname(__file__), '../vocabularies', 'data_vocab.vc')
        with open(vocab_saved_path, 'rb') as input:
            vocabulary_train = pickle.load(input)

        X = [[a for a in sentence if a in vocabulary_train] for sentence in X]
        sentences_padded_test = pad_sequences(X, maxlen=seq_length_max, dtype='str', padding = 'post', truncating ='post')
        sentences_padded_test = [sentence[:seq_length_max] + ['0'] * (seq_length_max - len(sentence)) for sentence in X]
        X = np.array([[vocabulary_train[word] for word in sentence] for sentence in sentences_padded_test])

        print("Data saved")

        return X
