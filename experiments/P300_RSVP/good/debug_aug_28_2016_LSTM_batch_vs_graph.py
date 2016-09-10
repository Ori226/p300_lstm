from keras.layers import BatchNormalization

from P300Net.data_preparation import create_data_rep_training, triplet_data_generator, triplet_data_collection, \
    triplet_data_generator_no_dict, get_number_of_samples_per_epoch, triplet_data_generator_no_dict_no_label, \
    get_number_of_samples_per_epoch_batch_mode, train_and_valid_generator
from P300Net.models import get_item_subgraph, get_graph, get_graph_lstm, get_item_lstm_subgraph
from experiments.P300_RSVP.common import *
from sklearn.utils import shuffle
import theano.tensor as T
import scipy

__author__ = 'ORI'

from sys import platform

is_local = True
if platform != "win32":
    is_local = False

import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt
from scipy import stats
import cPickle as pickle

rng = np.random.RandomState(42)
from os.path import basename

this_file_names = basename(__file__).split('.')[0]
print (this_file_names)


def create_training_and_testing_per_character(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                                              take_same_number_positive_and_negative=False):
    train_data, train_tags = create_train_data(gcd_res, fist_time_stamp, last_time_stamp, down_samples_param)
    test_data, test_tags = create_evaluation_data(gcd_res=gcd_res, fist_time_stamp=fist_time_stamp,
                                                  last_time_stamp=last_time_stamp,
                                                  down_samples_param=down_samples_param)
    func_args = dict(fist_time_stamp=fist_time_stamp, last_time_stamp=last_time_stamp,
                     down_samples_param=down_samples_param,
                     take_same_number_positive_and_negative=take_same_number_positive_and_negative)
    return train_data, train_tags, test_data, test_tags, func_args




def get_all_triplet_combinations(all_data_per_char, target_per_char):
    from itertools import combinations
    block_length = 10
    all_training_set = []

    number_of_repetition = 3
    magic_number = number_of_repetition * 30

    number_in_batch = 90
    all_data = np.zeros((magic_number, number_in_batch, all_data_per_char.shape[2], all_data_per_char.shape[3]),
                        dtype=np.float32)
    all_tags = np.zeros((number_in_batch, 30), dtype=np.int8)

    counter = 0
    for block_of_repetition_index in np.random.permutation(np.where(train_mode_per_block)[0].reshape(-1, 10))[
                                     0:10]:
        indexes = list(combinations(block_of_repetition_index, number_of_repetition))
        # now select 3 randomaly

        for index_i in np.random.permutation(indexes)[0:9]:
            print counter
            all_data[:, counter, :, :] = np.vstack([all_data_per_char[item] for item in index_i])
            all_tags[counter] = np.mean(np.vstack([target_per_char[item] for item in index_i]), axis=0)
            counter += 1

    return all_data, all_tags


def get_all_triplet_combinations_testing(all_data_per_char, target_per_char, train_mode_per_block):
    from itertools import combinations


    number_of_repetition = 3
    magic_number = number_of_repetition * 30

    number_in_batch = 40
    all_data = np.zeros((magic_number, number_in_batch, all_data_per_char.shape[2], all_data_per_char.shape[3]),
                        dtype=np.float32)
    all_tags = np.zeros((number_in_batch, 30), dtype=np.int8)


    counter = 0
    for block_of_repetition_index in np.random.permutation(np.where(train_mode_per_block != 1)[0].reshape(-1, 10))[
                                     0:10]:  # np.random.permutation(range(np.sum(train_mode_per_block == 1)/block_length))[0:10]:
        indexes = list(combinations(block_of_repetition_index, number_of_repetition))
        # now select 3 randomaly

        for index_i in np.random.permutation(indexes)[0:4]:
            all_data[:, counter, :, :] = np.vstack([all_data_per_char[item] for item in index_i])
            all_tags[counter] = np.mean(np.vstack([target_per_char[item] for item in index_i]), axis=0)
            counter += 1

    return all_data, all_tags

def predict_p300_model(model, train_data_as_matrix):
    # all_prediction = np.zeros((data_to_predict[0].shape[0], 30))
    flatten_data = train_data_as_matrix.reshape(train_data_as_matrix.shape[0] * train_data_as_matrix.shape[1],
                                                train_data_as_matrix.shape[2], train_data_as_matrix.shape[3])
    return model.predict(scipy.stats.zscore(flatten_data, axis=1))


def train_p300_model(model, train_data_as_matrix, train_tags_as_matrix):
    # flatten the data

    """
    @type model: Sequential
    """

    flatten_data = train_data_as_matrix.reshape(train_data_as_matrix.shape[0] * train_data_as_matrix.shape[1],
                                                train_data_as_matrix.shape[2], train_data_as_matrix.shape[3])
    flatten_tags = train_tags_as_matrix.reshape(train_tags_as_matrix.shape[0] * train_tags_as_matrix.shape[1], 1)
    shuffle_flatten_data, shuffle_flatten_tags, = shuffle(flatten_data, flatten_tags, random_state=0)

    model.fit(scipy.stats.zscore(shuffle_flatten_data, axis=1), shuffle_flatten_tags, nb_epoch=5,
              sample_weight=np.ones((shuffle_flatten_tags.shape[0])) + shuffle_flatten_tags.flatten() * 30)




from keras import backend as K





def identity_loss_v3(y_true, y_pred):
    import theano
    y_true_reshaped = K.mean(K.reshape(y_true, (-1, 3, 30)), axis=1)
    y_pred_reshaped = K.softmax(K.mean(K.reshape(y_pred, (-1, 3, 30)), axis=1))
    final_val = K.mean(K.categorical_crossentropy(y_pred_reshaped, y_true_reshaped))
    return final_val + y_pred * 0


if is_local:
    data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
    experiments_dir = r'C:\Users\ORI\Documents\Thesis\expreiment_results'

else:
    data_base_dir = r'/data_set'
    experiments_dir = r'/results'


def get_only_P300_model(eeg_sample_shape):

    digit_input = Input(shape=eeg_sample_shape)
    x = Flatten(input_shape=eeg_sample_shape)(digit_input)
    x = Dense(3)(x)
    x = Dense(3)(x)
    x = Dense(1)(x)
    out = Activation('tanh')(x)

    model = Model(digit_input, out)
    return model

def get_P300_model():
    number_of_timestamps = 25
    _num_of_hidden_units = 100

    eeg_sample_shape = (25, 55)
    model = get_only_P300_model(eeg_sample_shape)

    model.compile(optimizer='SGD',
                  loss=identity_loss_v3)
    return model


def get_model_by_graph(select=3,dictionary_size=30):
    from keras.layers import merge, Input, Dense, Flatten, Activation, Lambda
    from keras.models import Model


    # first, define the vision modules
    from keras import backend as K

    def per_char_loss(X):
        # alls = X.values()
        concatenated = X  # K.concatenate(alls)
        reshaped = K.mean(K.reshape(concatenated, (-1,select, 30)), axis=1)

        return reshaped  # K.softmax(K.reshape(reshaped, (reshaped.shape[0], -1)))

    eeg_sample_shape = (25,55)



    # P300 identification model
    p300_identification_model = get_only_P300_model(eeg_sample_shape) # Model(digit_input, out)

    all_inputs = [Input(shape=eeg_sample_shape) for _ in range(select*dictionary_size)]

    all_outs = [p300_identification_model(input_i) for input_i in all_inputs]

    concatenated = merge(all_outs, mode='concat')
    out = Lambda(per_char_loss, output_shape=(dictionary_size,))(concatenated)
    out = Activation('softmax')(out)
    # out = Dense(30)(concatenated)


    # out = Dense(1, activation='sigmoid')(concatenated)

    classification_model = Model(all_inputs, out)
    classification_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return classification_model, p300_identification_model



def convert_batch_input_to_graph(batch_input):
    pass

if __name__ == "__main__":



    all_subjects = [
                    "RSVP_Color116msVPicn.mat"];

    for experiment_counter, subject in enumerate(all_subjects):
        # subject = "RSVP_Color116msVPgcd.mat"

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800, downsampe_params=8)



        # data_generator = triplet_data_generator_no_dict(all_data_per_char_as_matrix[train_mode_per_block == 1],
        #                                                 target_per_char_as_matrix[train_mode_per_block == 1], 40, 3)


        data_generator_batch = triplet_data_generator_no_dict(all_data_per_char_as_matrix[train_mode_per_block == 1],
                                                        target_per_char_as_matrix[train_mode_per_block == 1], 1, 3,debug_mode=True)

        data_for_batch = data_generator_batch.next()

        data_generator, valid_generator_all, traininig_size, validation_size = train_and_valid_generator(
            all_data_per_char_as_matrix[train_mode_per_block == 1],
            target_per_char_as_matrix[train_mode_per_block == 1], batch_size=1, select=3, return_dict=False,debug_mode=True)

        data_for_graph = data_generator.next()
        number_of_samples_in_epoch = get_number_of_samples_per_epoch_batch_mode(
            all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0], 3, 10)


        total_number_of_char_in_training = 24000  # all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0]/10

        from keras.models import Sequential
        from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
        from keras.layers.recurrent import LSTM, GRU
        from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, Lambda
        from keras.models import Model
        from keras.regularizers import l2



        #
        # model.add(LSTM(input_dim=55, output_dim=_num_of_hidden_units, input_length=number_of_timestamps,
        #                return_sequences=True))
        # model.add(LSTM(input_dim=_num_of_hidden_units, output_dim=_num_of_hidden_units, return_sequences=False))
        # model.add(Dense(1, input_dim=1, activation='sigmoid'))



        print "after compile"
        X,y = data_generator.next()
        model = get_P300_model()
        # model_training_results = model.train_on_batch(X,y)
        loss_function = model.get_loss(X, y)
        # loss_function(X, y)
        model_training_results = model.get_batch_gradient(X, y)


        model_training_results = model.fit_generator(data_generator, number_of_samples_in_epoch, nb_epoch=2,
                                                     max_q_size=1)
        final_model = model

        all_prediction_P300Net = predict_p300_model(final_model, all_data_per_char_as_matrix[train_mode_per_block != 1])

        x = T.dmatrix('x')
        import theano

        softmax_res_func = theano.function([x], T.nnet.softmax(x))

        test_tags = target_per_char_as_matrix[
            train_mode_per_block != 1]  # np.array([target_per_char[x][train_mode_per_block != 1] for x in range(30)]).T
        all_res = test_tags

        actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
        gt = np.argmax(np.mean(all_res.reshape((-1, 10, 30)), axis=1), axis=1)
        accuracy = np.sum(actual == gt) / float(len(gt))

        print "subject:{0} accu:{1}".format(subject, accuracy)
        all_model_results = dict(subject=subject, accuracy=accuracy, prediction_results=all_prediction_P300Net)

        pickle.dump(all_model_results, file=open(
            os.path.join(experiments_dir, this_file_names + "_" + str(experiment_counter) + "_" + subject) + ".p",
            "wb"))

        # save: loss historay, final accuracy, model weight ( in order to achieve TP\FP, source data
