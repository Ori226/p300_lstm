from experiments.P300_RSVP.common import *

__author__ = 'ORI'

import numpy as np
import os
import sklearn
import cPickle as pickle
# I should learn how to load libraries in a more elegant way


import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
# from OriKerasExtension import ThesisHelper

# reload(OriKerasExtension)
# reload(OriKerasExtension.ThesisHelper)
from   OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
import OriKerasExtension.P300Prediction

reload(OriKerasExtension.P300Prediction)

# import OriKerasExtension
#
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from keras.layers.recurrent import LSTM
# from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

# reload(OriKerasExtension)


rng = np.random.RandomState(42)


#
# def create_evaluation_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1):
#     #     gcd_res = readCompleteMatFile(file_name)
#     data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'], gcd_res['target'],
#                                     fist_time_stamp, last_time_stamp)
#     # print  data_for_eval
#
#     temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)
#
#     test_data_gcd, test_target_gcd = temp_data_for_eval[gcd_res['train_mode'] != 1], data_for_eval[1][
#         gcd_res['train_mode'] != 1]
#     return test_data_gcd, test_target_gcd
#
#
# def downsample_data(data, number_of_original_samples, down_samples_param):
#     new_number_of_time_stamps = number_of_original_samples / down_samples_param
#
#     # print  data_for_eval
#     temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))
#
#     for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
#         temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
#     return temp_data_for_eval
#
#
# def create_training_and_testing(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
#                                 take_same_number_positive_and_negative=False):
#     train_data, train_tags = create_train_data(gcd_res, fist_time_stamp, last_time_stamp, down_samples_param,
#                                                take_same_number_positive_and_negative)
#     test_data, test_tags = create_evaluation_data(gcd_res=gcd_res, fist_time_stamp=fist_time_stamp,
#                                                   last_time_stamp=last_time_stamp,
#                                                   down_samples_param=down_samples_param)
#     func_args = dict(fist_time_stamp=fist_time_stamp, last_time_stamp=last_time_stamp,
#                      down_samples_param=down_samples_param,
#                      take_same_number_positive_and_negative=take_same_number_positive_and_negative)
#     return train_data, train_tags, test_data, test_tags, func_args
#
#
# def create_train_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
#                       take_same_number_positive_and_negative=False):
#     all_positive_train = []
#     all_negative_train = []
#
#     data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
#                                     gcd_res['target'], fist_time_stamp, last_time_stamp)
#
#     temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)
#
#     # # extract the calibration_data
#     # positive_train_data_gcd = temp_data_for_eval[
#     #     np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 1], axis=0)]
#     # negative_train_data_gcd = temp_data_for_eval[
#     #     np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 0], axis=0)]
#
#     all_tags = gcd_res['target'][gcd_res['train_mode'] == 1]
#
#
#     # all_positive_train.append(positive_train_data_gcd)
#     # all_negative_train.append(negative_train_data_gcd)
#
#     all_data = temp_data_for_eval[gcd_res['train_mode'] == 1]
#     # if take_same_number_positive_and_negative:
#     #     negative_train_data_gcd = rng.permutation(np.vstack(all_negative_train))[0:positive_train_data_gcd.shape[0]]
#     # else:
#     #     negative_train_data_gcd = np.vstack(all_negative_train)
#     #
#     # all_data = np.vstack([positive_train_data_gcd, negative_train_data_gcd])
#
#     # all_tags = np.vstack(
#     #     [np.ones((positive_train_data_gcd.shape[0], 1)), np.zeros((negative_train_data_gcd.shape[0], 1))]).astype(np.int)
#     categorical_tags = to_categorical(all_tags)
#
#     # shuffeled_samples, suffule_tags = shuffle(all_data, all_tags, random_state=0)
#
#
#     return all_data, all_tags

#
# def create_data_for_compare_by_repetition(file_name):
#     gcd_res = readCompleteMatFile(file_name)
#     sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
#                        train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
#                        stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])
#     return sub_gcd_res


# class run_experiments(object):
#     def get_training_data(self,
#                           start_time,
#                           end_time,
#                           downsample_rate=1,
#                           negatvie_percent=1.0):
#         pass
#
#
#
#     def evaluate_on_model():
#         pass

# from sklearn.lda import LDA

#
# class LSTM_CNN_EEG_Comb(GeneralModel):
#     def get_params(self):
#         super(LSTM_CNN_EEG_Comb, self).get_params()
#         return self.model.get_weights()
#
#     def get_name(self):
#         super(LSTM_CNN_EEG_Comb, self).get_name()
#         return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)
#
#     def reset(self):
#         super(LSTM_CNN_EEG_Comb, self).reset()
#         self.model.set_weights(self.original_weights)
#
#     def __init__(self, positive_weight, _num_of_hidden_units):
#         super(LSTM_CNN_EEG_Comb, self).__init__()
#         self.positive_weight = positive_weight
#         self._num_of_hidden_units = _num_of_hidden_units
#
#         '''
#         define the neural network model:
#
#         '''
#         # from keras.layers.extra import *
#
#         from keras.models import Sequential
#         # from keras.initializations import norRemal, identity
#         from keras.layers.recurrent import LSTM
#         from keras.layers.convolutional import Convolution2D
#         from keras.layers.core import Dense, Activation, Reshape
#         # from keras.layers.wrappers import TimeDistributed
#         from keras.layers.convolutional import MaxPooling2D
#         from keras.layers.core import Permute
#
#         from keras.regularizers import l2
#
#         maxToAdd = 200
#         # define our time-distributed setup
#         model = Sequential()
#
#         model.add(Reshape((1, maxToAdd, 55), input_shape=(maxToAdd, 55)))  # this line updated to work with keras 1.0.2
#         # model.add(TimeDistributedDense(10, input_shape=(maxToAdd, 55)))
#         model.add(Convolution2D(3, 12, 55, border_mode='valid', W_regularizer=l2(0.1)))  # org
#         model.add(Activation('tanh'))
#         model.add(MaxPooling2D(pool_size=(12, 1), border_mode='valid'))
#         model.add(Permute((2, 1, 3)))
#         model.add(Reshape((model.layers[-1].output_shape[1],
#                            model.layers[-1].output_shape[2])))  # this line updated to work with keras 1.0.2
#         model.add(LSTM(output_dim=10, return_sequences=False))
#         #
#         model.add(Dense(2, activation='softmax'))
#
#
#         model.compile(optimizer='rmsprop',
#                       loss='categorical_crossentropy')
#         self.model = model
#
#         # model.predict(np.random.rand(28, 200, 55).astype(np.float32)).shape
#
#         print model.layers[-1].output_shape
#         # print "2 {} {}".format(model.layers[1].output_shape[-3:], (1, maxToAdd, np.prod(model.layers[1].output_shape[-3:])))
#         self.original_weights = self.model.get_weights()
#         """ :type Sequential"""
#
#     def fit(self, _X, y):
#         from keras.callbacks import ModelCheckpoint
#
#         _y = to_categorical(y)
#         # _X = np.expand_dims(np.expand_dims(_X,3),4).transpose([0,1,3,2,4])
#
#
#         # (batch, times, color_channel, x, y)
#
#         checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)
#         sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))
#
#         self.model.fit(stats.zscore(_X[sss[0][0]], axis=1), _y[sss[0][0]],
#                        nb_epoch=30, show_accuracy=True, verbose=1,
#                        validation_data=(
#                            stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
#                        class_weight={0: 1, 1: self.positive_weight},
#                        callbacks=[checkpointer])
#
#
#     def predict(self, _X):
#         return self.model.predict(stats.zscore(_X, axis=1))


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


from keras.layers.core import Dense


def get_item_subgraph(input_shape, latent_dim):
    # Could take item metadata here, do convolutional layers etc.

    # model = Sequential()
    # model.add(Dense(latent_dim, input_shape=input_shape))



    from keras.models import Sequential
    from keras.layers.convolutional import Convolution2D
    from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
    from keras.layers.convolutional import MaxPooling2D

    from keras.regularizers import l2
    number_of_time_stamps = 200
    number_of_in_channels = 55
    number_of_out_channels = 10
    model = Sequential()

    model.add(Flatten(input_shape=(number_of_time_stamps, number_of_in_channels)))
    model.add(Dense(40))  # , input_shape=(number_of_time_stamps, number_of_in_channels)))
    model.add(Dropout(0.1))  # , input_shape=(number_of_time_stamps, number_of_in_channels)))
    model.add(Activation('tanh'))
    model.add(Dense(40))
    model.add(Dropout(0.1))  # , input_shape=(number_of_time_stamps, number_of_in_channels)))
    model.add(Activation('tanh'))
    model.add(Dense(1))

    return model


def get_user_subgraph(input_shape, latent_dim):
    # Could do all sorts of fun stuff here that takes
    # user metadata in.

    model = Sequential()
    model.add(Dense(latent_dim, input_shape=input_shape))

    return model


from keras.layers.core import Dense, Lambda
from keras import backend as K
from keras.models import Sequential, Graph


def bpr_triplet_loss(X):
    user_latent, item_latent = X.values()
    positive_item_latent, negative_item_latent = item_latent.values()

    # BPR loss
    loss = - K.sigmoid(K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True)
                       - K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss


def per_char_loss(X):
    # pos_lat, neg_lat = X.values()
    alls = X.values()
    # positive_item_latent, negative_item_latent = item_latent.values()

    # BPR loss

    # K.argmax(K.concatenate([pos_lat, neg_lat]))
    # loss = K.max(K.concatenate([pos_lat, neg_lat]))
    concatenated = K.concatenate(alls)
    # reshaped = K.argmax(K.sum(K.reshape(concatenated, (K.shape(concatenated)[0],2, 30)),axis=1),axis=1)
    reshaped = K.mean(K.reshape(concatenated, (K.shape(concatenated)[0], 3, 30)), axis=1)

    return K.softmax(K.reshape(reshaped, (reshaped.shape[0], -1)))


def identity_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)  # K.mean(y_pred - 0 * y_true)


def get_graph(num_items, latent_dim):
    batch_input_shape = (1, 200, 55)
    # batch_input_shape = None
    magic_num = 90
    model = Graph()

    # Add inputs
    # model.add_input('user_input', input_shape=(num_users,), batch_input_shape=batch_input_shape)
    for i in range(magic_num):
        model.add_input('positive_item_input_{}'.format(i), input_shape=(200, 55), batch_input_shape=batch_input_shape)
    # model.add_input('negative_item_input', input_shape=(200,55), batch_input_shape=batch_input_shape)

    # Add shared-weight item subgraph

    # input_dict = dict(["positive_item_input_{}".format(i) for i in range(5)])

    model.add_shared_node(get_item_subgraph((num_items,), latent_dim),
                          name='item_latent',
                          inputs=["positive_item_input_{}".format(i) for i in range(magic_num)],
                          merge_mode='join')
    # # Add user embedding
    # model.add_node(get_user_subgraph((num_users,), latent_dim),
    #                name='user_latent',
    #                input='user_input')

    # Compute loss
    model.add_node(Lambda(per_char_loss),
                   name='triplet_loss',
                   input='item_latent')

    # Add output
    model.add_output(name='triplet_loss', input='triplet_loss')

    # Compile using a dummy loss to fit within the Keras paradigm
    # model.compile(loss={'triplet_loss': identity_loss}, optimizer='sgd')#Adagrad(lr=0.1, epsilon=1e-06))
    model.compile(loss={'triplet_loss': identity_loss}, optimizer='sgd')  # Adagrad(lr=0.1, epsilon=1e-06))

    return model


def bpr_triplet_loss(X):
    user_latent, item_latent = X.values()
    positive_item_latent, negative_item_latent = item_latent.values()

    # BPR loss
    loss = - K.sigmoid(K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True)
                       - K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss


def get_all_triplet_combinations_indexes_only(all_data_per_char, target_per_char):
    from itertools import combinations
    block_length = 10
    all_training_set = []

    number_of_repetition = 3
    magic_number = number_of_repetition * 30

    # number_in_batch = 90
    # all_data = np.zeros((magic_number, number_in_batch, 200, 55), dtype=np.float32)
    # all_tags = np.zeros((number_in_batch, 30), dtype=np.int8)
    # data_in_dict = dict(
    #     [["positive_item_input_{}".format(i), np.zeros((60, number_in_batch,200,55))] for i in
    #      range(60)])

    counter = 0
    all_combination = []
    for block_of_repetition_index in np.where(train_mode_per_block)[0].reshape(-1, 10):  # np.random.permutation(range(np.sum(train_mode_per_block == 1)/block_length))[0:10]:
        indexes = list(combinations(block_of_repetition_index, number_of_repetition))
        # now select 3 randomaly
        all_combination.extend(indexes)
        # for index_i in np.random.permutation(indexes)[0:9]:
        #     all_data[:, counter, :, :] = np.vstack([all_data_per_char[item] for item in index_i])
        #     all_tags[counter] = np.mean(np.vstack([target_per_char[item] for item in index_i]), axis=0)
        #     counter += 1

    return all_combination


def get_all_triplet_combinations(all_data_per_char, target_per_char):
    from itertools import combinations
    block_length = 10
    all_training_set = []

    number_of_repetition = 3
    magic_number = number_of_repetition * 30

    number_in_batch = 90
    all_data = np.zeros((magic_number, number_in_batch, 200, 55), dtype=np.float32)
    all_tags = np.zeros((number_in_batch, 30), dtype=np.int8)
    # data_in_dict = dict(
    #     [["positive_item_input_{}".format(i), np.zeros((60, number_in_batch,200,55))] for i in
    #      range(60)])

    counter = 0
    for block_of_repetition_index in np.random.permutation(np.where(train_mode_per_block)[0].reshape(-1, 10))[
                                     0:10]:  # np.random.permutation(range(np.sum(train_mode_per_block == 1)/block_length))[0:10]:
        indexes = list(combinations(block_of_repetition_index, number_of_repetition))
        # now select 3 randomaly

        for index_i in np.random.permutation(indexes)[0:9]:
            all_data[:, counter, :, :] = np.vstack([all_data_per_char[item] for item in index_i])
            all_tags[counter] = np.mean(np.vstack([target_per_char[item] for item in index_i]), axis=0)
            counter += 1

    return all_data, all_tags


def get_all_triplet_combinations_testing(all_data_per_char, target_per_char, train_mode_per_block):
    from itertools import combinations
    block_length = 10
    all_training_set = []

    number_of_repetition = 3
    magic_number = number_of_repetition * 30

    number_in_batch = 40
    all_data = np.zeros((magic_number, number_in_batch, 200, 55), dtype=np.float32)
    all_tags = np.zeros((number_in_batch, 30), dtype=np.int8)
    # data_in_dict = dict(
    #     [["positive_item_input_{}".format(i), np.zeros((60, number_in_batch,200,55))] for i in
    #      range(60)])

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


def create_data_rep_training(file_name, fist_time_stamp, last_time_stamp):
    gcd_res = readCompleteMatFile(file_name)
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    data_for_eval = (downsample_data(data_for_eval[0], data_for_eval[0].shape[1], 1), data_for_eval[1])

    train_mode_per_block = gcd_res['train_mode'].reshape(-1, 30)[:, 0]
    all_data_per_char_as_matrix = np.zeros(
        (train_mode_per_block.shape[0], 30, data_for_eval[0].shape[1], data_for_eval[0].shape[2]))
    all_data_per_char = dict()
    target_per_char_as_matrix = np.zeros((train_mode_per_block.shape[0], 30), dtype=np.int)
    for i, stimuli_i in enumerate(range(1, 31)):
        all_data_per_char[i] = data_for_eval[0][gcd_res['stimulus'] == stimuli_i]
        all_data_per_char_as_matrix[:, i, :, :] = data_for_eval[0][gcd_res['stimulus'] == stimuli_i]

    target_per_char = dict()
    for i, stimuli_i in enumerate(range(1, 31)):
        target_per_char[i] = data_for_eval[1][gcd_res['stimulus'] == stimuli_i]
        target_per_char_as_matrix[:, i] = data_for_eval[1][gcd_res['stimulus'] == stimuli_i]

    return all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix


def predict_p300_model(model, data_to_predict):
    all_prediction = np.zeros((data_to_predict[0].shape[0], 30))
    for stimuli_i in range(30):
        all_prediction[:, stimuli_i] = model.predict(
            stats.zscore(data_to_predict[stimuli_i], axis=1)).flatten()
    return all_prediction


def train_p300_model(model, train_data_as_matrix, train_tags_as_matrix):
    # flatten the data
    from sklearn.utils import shuffle

    flatten_data = train_data_as_matrix.reshape(train_data_as_matrix.shape[0] * train_data_as_matrix.shape[1],
                                                train_data_as_matrix.shape[2], train_data_as_matrix.shape[3])
    flatten_tags = train_tags_as_matrix.reshape(train_tags_as_matrix.shape[0] * train_tags_as_matrix.shape[1], 1)
    shuffle_flatten_data, shuffle_flatten_tags, = shuffle(flatten_data, flatten_tags, random_state=0)
    model.fit(shuffle_flatten_data, shuffle_flatten_tags, nb_epoch=20,
              sample_weight=np.ones((shuffle_flatten_tags.shape[0])) + shuffle_flatten_tags.flatten() * 50)
    print "temp"
    pass


def create_train_test_validation(all_data_per_char_as_matrix, target_per_char_as_matrix):
    total_number_of_repetition = 10
    train_data = all_data_per_char_as_matrix[train_mode_per_block == 1]
    train_tags = target_per_char_as_matrix[train_mode_per_block == 1]
    total_number_of_char_in_training = train_data.shape[0] / total_number_of_repetition

    # extract validation set indexes:
    validation_indexes = list(sklearn.cross_validation.ShuffleSplit(total_number_of_char_in_training, n_iter=1, test_size=0.2,
                                               train_size=0.8, random_state=0))

    print "temp"


data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
if __name__ == "__main__":

    all_subjects = ["RSVP_Color116msVPpia.mat",
                    "RSVP_Color116msVPicr.mat",
                    "RSVP_Color116msVPfat.mat",
                    "RSVP_Color116msVPgcb.mat",
                    "RSVP_Color116msVPgcc.mat",
                    "RSVP_Color116msVPgcd.mat",
                    "RSVP_Color116msVPgcf.mat",
                    "RSVP_Color116msVPgcg.mat",
                    "RSVP_Color116msVPgch.mat",
                    "RSVP_Color116msVPiay.mat",
                    "RSVP_Color116msVPicn.mat"];

    for subject in all_subjects:
        # subject = "RSVP_Color116msVPgcd.mat"

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800)

        res = get_all_triplet_combinations_indexes_only(all_data_per_char_as_matrix,
                                                        target_per_char_as_matrix)

        temp = create_train_test_validation(all_data_per_char_as_matrix, target_per_char_as_matrix)

        # all_data = create_data_for_compare_by_repetition(file_name)

        # get_all_triplet_combinations(all_data_per_char, target_per_char)
        # extract % of the data for validation. it should contains several 10 repetition

        total_number_of_char_in_training = all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0]/10

        sklearn.cross_validation.ShuffleSplit(len(total_number_of_char_in_training), n_iter=1, test_size=0.1, train_size=0.9, random_state=0)
        list(sklearn.cross_validation.ShuffleSplit(total_number_of_char_in_training, n_iter=1, test_size=0.2,
                                                   train_size=0.8, random_state=0))

        res = get_all_triplet_combinations_indexes_only(all_data_per_char_as_matrix,
                                                                          target_per_char_as_matrix)
        import sklearn


        testing_data, testing_tags = get_all_triplet_combinations_testing(all_data_per_char_as_matrix,
                                                                          target_per_char_as_matrix,
                                                                          train_mode_per_block)

        # gcd_res = readCompleteMatFile(file_name)
        # repetition_eval = EvaluateByRepetition(file_name)

        # training_data, train_tags, testing_data, test_tags, func_args = create_training_and_testing(gcd_res, -200, 800, 1, False)

        # input_dict = dict(
        #     [["positive_item_input_{}".format(i), stats.zscore(training_data[30 * (i ):30 * (i + 2):30], axis=1)] for i in
        #      range(60)])



        # input_dict_testing = dict(
        #     [["positive_item_input_{}".format(i), stats.zscore(testing_data[i], axis=1)] for i in
        #      range(90)])
        #
        # input_dict_testing['triplet_loss'] = testing_tags

        model = get_graph(3, 10)
        temp_model = get_item_subgraph(None, None)
        temp_model.compile(loss='binary_crossentropy', class_mode="binary", optimizer='sgd')
        all_train_data = dict()
        train_p300_model(temp_model, all_data_per_char_as_matrix[train_mode_per_block == 1],
                         target_per_char_as_matrix[train_mode_per_block == 1])

        for x in range(30):
            all_train_data[x] = all_data_per_char[x][train_mode_per_block == 1]

        all_test_data = dict()
        for x in range(30):
            all_test_data[x] = all_data_per_char[x][train_mode_per_block != 1]

        # predict_p300_model(temp_model, all_train_data)

        for epoch in range(2):
            print "starting {0}".format(epoch)
            training_data, train_tags = get_all_triplet_combinations(
                all_data_per_char_as_matrix[train_mode_per_block == 1],
                target_per_char_as_matrix[train_mode_per_block == 1],
                train_mode_per_block)
            input_dict = dict(
                [["positive_item_input_{}".format(i), stats.zscore(training_data[i], axis=1)] for i in
                 range(90)])

            input_dict['triplet_loss'] = train_tags
            model.train_on_batch(input_dict)
        final_model = get_item_subgraph(None, None)
        final_model_original_weights = final_model.get_weights()
        temp_weight = list(model.nodes['item_latent'].layer.get_weights())
        final_model.compile(loss='binary_crossentropy', class_mode="binary", optimizer='sgd')
        final_model.set_weights(temp_weight)
        import theano.tensor as T

        all_prediction = np.zeros((all_data_per_char[0][train_mode_per_block != 1].shape[0], 30))
        for stimuli_i in range(30):
            all_prediction[:, stimuli_i] = final_model.predict(
                stats.zscore(all_data_per_char[stimuli_i][train_mode_per_block != 1], axis=1)).flatten()
            # input_dict = dict([["positive_item_input_{}".format(i),np.random.rand(1,55,200).astype(np.float32)] for i in range(60)])

        all_prediction_untrained = predict_p300_model(temp_model, all_test_data)
        plt.subplot(1, 4, 1)
        plt.imshow(all_prediction, interpolation='none')
        plt.subplot(1, 4, 2)
        x = T.dmatrix('x')
        import theano

        softmax_res_func = theano.function([x], T.nnet.softmax(x))

        # softmax_res = softmax_res_func(all_res)
        # plt.imshow(softmax_res, interpolation='none')
        # plt.show()
        plt.imshow(softmax_res_func(all_prediction), interpolation='none')
        plt.subplot(1, 4, 3)
        plt.imshow(softmax_res_func(np.mean(all_prediction.reshape((-1, 10, 30)), axis=1)).astype(np.int),
                   interpolation='none')

        plt.subplot(1, 4, 4)
        all_res = np.array([target_per_char[x][train_mode_per_block != 1] for x in range(30)]).T
        plt.imshow(np.mean(all_res.reshape((-1, 10, 30)), axis=1), interpolation='none')
        # plt.show()

        actual_untrained = \
        np.where(np.round(softmax_res_func(np.mean(all_prediction_untrained.reshape((-1, 10, 30)), axis=1))) == 1)[0];
        actual = np.where(np.round(softmax_res_func(np.mean(all_prediction.reshape((-1, 10, 30)), axis=1))) == 1)[0];
        gt = np.where(np.mean(all_res.reshape((-1, 10, 30)), axis=1) == 1)[0]
        np.intersect1d(actual, gt)
        accuracy = len(np.intersect1d(actual, gt)) / float(len(gt))
        accuracy_untrained = len(np.intersect1d(actual_untrained, gt)) / float(len(gt))
        print "subject:{0} accu:{1} acc_untrained{2}".format(subject, accuracy, accuracy_untrained)
        # target_per_char_as_matrix[0:10,:]

    # prediction_res = model.predict(input_dict_testing)
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(prediction_res['triplet_loss'], interpolation='none')
    # plt.subplot(1, 3, 2)
    #
    # plt.plot(1 - scipy.stats.entropy(prediction_res['triplet_loss'].T).reshape(-1, 1))
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(testing_tags, interpolation='none')
    # plt.tight_layout()
    # plt.show()

    # print "temp"
    #
    #
    # # model.fit({
    # #            'positive_item_input': np.random.rand(3,55,200).astype(np.float32),
    # #     'negative_item_input': np.random.rand(3,55,200).astype(np.float32),
    # #             'triplet_loss': np.random.rand(3,2).astype(np.float32)},
    # #           batch_size=1,
    # #           nb_epoch=1,
    # #           verbose=2,
    # #           shuffle=True)
    #
    # model_20 = None
    # model_100 = None
    #
    # all_subjects = ["RSVP_Color116msVPpia.mat",
    #     "RSVP_Color116msVPicr.mat",
    #                 "RSVP_Color116msVPfat.mat",
    #                 "RSVP_Color116msVPgcb.mat",
    #                 "RSVP_Color116msVPgcc.mat",
    #                 "RSVP_Color116msVPgcd.mat",
    #                 "RSVP_Color116msVPgcf.mat",
    #                 "RSVP_Color116msVPgcg.mat",
    #                 "RSVP_Color116msVPgch.mat",
    #                 "RSVP_Color116msVPiay.mat",
    #                 "RSVP_Color116msVPicn.mat"];
    #
    # # all_subjects = ["RSVP_Color116msVPicr.mat"]
    #
    #
    # # model = LDA()
    #
    # all_models = [LSTM_CNN_EEG_Comb(50, 20)]
    # for model_type in all_models:
    #
    #     all_model_results = []
    #
    #     for subject in all_subjects:
    #         file_name = os.path.join(data_base_dir, subject)
    #         gcd_res = readCompleteMatFile(file_name)
    #         repetition_eval = EvaluateByRepetition(file_name)
    #
    #         for data_extraction_method in [create_training_and_testing(gcd_res, -200, 800, 1, False)
    #                                        ]:
    #
    #             # create_training_and_testing(gcd_res, 0, 400, 1, True)
    #             training_data, train_tags, testing_data, test_tags, func_args = data_extraction_method
    #             model = model_type  # type: GeneralModel
    #             print "starting {}:{}:{}".format(subject, model.get_name()[-7:-4], ",".join([str(x) for x in func_args.values()]))
    #
    #             # training_data = create_train_data(gcd_res, 0, 400, 1, True)
    #             # testing_data = create_evaluation_data(gcd_res, 0, 400, 1)
    #
    #             # training_data, train_tags, testing_data, test_tags = create_training_and_testing(gcd_res, 0, 400, 1, True)
    #
    #             # model = My_LDA()
    #             model.fit(training_data, train_tags)
    #             prediction_res = model.predict(testing_data)
    #             all_accuracies = repetition_eval.foo(test_tags, prediction_res)
    #             all_model_results.append(
    #                 dict(all_accuracies=all_accuracies, subject_name=subject, model=model.get_name(),
    #                      model_params=model.get_params(), func_args=func_args))
    #             model.reset()
    #             print "end {}:{}:{}".format(subject, model.get_name()[-7:-4],
    #                                              ",".join([str(x) for x in func_args.values()]))
    #
    #
    #
    #
    #     pickle.dump(all_model_results, file=open(model_type.get_name() + "_b.p", "wb"))
    #
    #
    # pass
