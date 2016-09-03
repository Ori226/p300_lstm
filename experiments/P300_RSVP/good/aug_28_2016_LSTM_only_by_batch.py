from keras.layers import BatchNormalization

from P300Net.data_preparation import create_data_rep_training, triplet_data_generator, triplet_data_collection, \
    triplet_data_generator_no_dict, get_number_of_samples_per_epoch, triplet_data_generator_no_dict_no_label
from P300Net.models import get_item_subgraph, get_graph,get_graph_lstm,  get_item_lstm_subgraph
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


# def get_all_triplet_combinations_indexes_only(all_data_per_char, target_per_char):
#     from itertools import combinations
#     block_length = 10
#     all_training_set = []
#
#     number_of_repetition = 3
#     magic_number = number_of_repetition * 30
#
#
#
#     counter = 0
#     all_combination = []
#     for block_of_repetition_index in np.where(train_mode_per_block)[0].reshape(-1, 10):
#         indexes = list(combinations(block_of_repetition_index, number_of_repetition))
#         # now select 3 randomaly
#         all_combination.extend(indexes)
#
#     return all_combination


def get_all_triplet_combinations(all_data_per_char, target_per_char):
    from itertools import combinations
    block_length = 10
    all_training_set = []

    number_of_repetition = 3
    magic_number = number_of_repetition * 30

    number_in_batch = 90
    all_data = np.zeros((magic_number, number_in_batch, all_data_per_char.shape[2], all_data_per_char.shape[3]), dtype=np.float32)
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
    block_length = 10
    all_training_set = []

    number_of_repetition = 3
    magic_number = number_of_repetition * 30

    number_in_batch = 40
    all_data = np.zeros((magic_number, number_in_batch, all_data_per_char.shape[2], all_data_per_char.shape[3]), dtype=np.float32)
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


# def predict_p300_model(model, data_to_predict):
#     all_prediction = np.zeros((data_to_predict[0].shape[0], 30))
#     for stimuli_i in range(30):
#         all_prediction[:, stimuli_i] = model.predict(
#             stats.zscore(data_to_predict[stimuli_i], axis=1)).flatten()
#     return all_prediction


def predict_p300_model(model, train_data_as_matrix):
    # all_prediction = np.zeros((data_to_predict[0].shape[0], 30))
    flatten_data = train_data_as_matrix.reshape(train_data_as_matrix.shape[0] * train_data_as_matrix.shape[1],
                                                train_data_as_matrix.shape[2], train_data_as_matrix.shape[3])
    return model.predict(scipy.stats.zscore(flatten_data, axis=1))

    # for stimuli_i in range(30):
    #     all_prediction[:, stimuli_i] = model.predict(
    #         stats.zscore(data_to_predict[stimuli_i], axis=1)).flatten()
    # return all_prediction


def train_p300_model(model, train_data_as_matrix, train_tags_as_matrix):
    # flatten the data

    """
    @type model: Sequential
    """

    flatten_data = train_data_as_matrix.reshape(train_data_as_matrix.shape[0] * train_data_as_matrix.shape[1],
                                                train_data_as_matrix.shape[2], train_data_as_matrix.shape[3])
    flatten_tags = train_tags_as_matrix.reshape(train_tags_as_matrix.shape[0] * train_tags_as_matrix.shape[1], 1)
    shuffle_flatten_data, shuffle_flatten_tags, = shuffle(flatten_data, flatten_tags, random_state=0)

    model.fit(scipy.stats.zscore(shuffle_flatten_data,axis=1), shuffle_flatten_tags, nb_epoch=5,
              sample_weight=np.ones((shuffle_flatten_tags.shape[0])) + shuffle_flatten_tags.flatten() * 30)


#
# def create_train_test_validation(all_data_per_char_as_matrix, target_per_char_as_matrix):
#     total_number_of_repetition = 10
#     train_data = all_data_per_char_as_matrix[train_mode_per_block == 1]
#     train_tags = target_per_char_as_matrix[train_mode_per_block == 1]
#     total_number_of_char_in_training = train_data.shape[0] / total_number_of_repetition
#
#     # extract validation set indexes:
#     validation_indexes = list(sklearn.cross_validation.ShuffleSplit(total_number_of_char_in_training, n_iter=1, test_size=0.2,
#                                                train_size=0.8, random_state=0))
#
#     print "temp"

from keras import backend as K
def identity_loss_v2(y_true, y_pred):

    # calculate the average loss over the whole batch:
    # K.reshape(dictionary_size*number_of_repetition, y_pred)
    # return K.mean(K.flatten(y_pred), axis=0)*0+ K.mean(y_pred,axis=1).shape[0]+0*y_pred.shape[0] - 0 * K.max(K.flatten(y_true))

    reshaped = K.mean(K.reshape(y_pred, (30, 3,-1)), axis=1)

    return K.mean(K.softmax(K.reshape(reshaped, (reshaped.shape[0], -1)))) +0*y_pred




def identity_loss_v3(y_true, y_pred):
    import theano
    y_true_reshaped = K.mean(K.reshape(y_true, (-1, 9, 30)), axis=1)
    y_pred_reshaped = K.softmax(K.mean(K.reshape(y_pred, (-1, 9, 30)), axis=1))


    # y_true_reshaped = K.reshape(y_true, (-1, 30))
    # x_printed = theano.printing.Print('this is a very important value')(y_true_reshaped)
    # y_pred_reshaped = K.softmax(K.reshape(y_pred, (-1, 30)))

    final_val = K.mean(K.categorical_crossentropy(y_pred_reshaped, y_true_reshaped))

    # final_val = K.mean(K.square(y_pred_reshaped- y_true_reshaped))

    return  final_val+y_pred*0

# C:\Users\ORI\Documents\Thesis\experiment_results
# C:\Users\ORI\Documents\Thesis\expreiment_results
if is_local:
    data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
    experiments_dir = r'C:\Users\ORI\Documents\Thesis\expreiment_results'

else:
    data_base_dir = r'/data_set'
    experiments_dir = r'/results'
if __name__ == "__main__":

    all_subjects = ["RSVP_Color116msVPicr.mat",
        "RSVP_Color116msVPpia.mat",
                    "RSVP_Color116msVPfat.mat",
                    "RSVP_Color116msVPgcb.mat",
                    "RSVP_Color116msVPgcc.mat",
                    "RSVP_Color116msVPgcd.mat",
                    "RSVP_Color116msVPgcf.mat",
                    "RSVP_Color116msVPgcg.mat",
                    "RSVP_Color116msVPgch.mat",
                    "RSVP_Color116msVPiay.mat",
                    "RSVP_Color116msVPicn.mat"];

    all_subjects = [
                    "RSVP_Color116msVPicn.mat"];

    for experiment_counter, subject in enumerate(all_subjects):
        # subject = "RSVP_Color116msVPgcd.mat"

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800,downsampe_params=8)

        # data_generator = triplet_data_generator(all_data_per_char_as_matrix[train_mode_per_block == 1], target_per_char_as_matrix[train_mode_per_block == 1], 80)

        data_generator = triplet_data_generator_no_dict(all_data_per_char_as_matrix[train_mode_per_block == 1], target_per_char_as_matrix[train_mode_per_block == 1], 6,9)
        # data_generator_no_label = triplet_data_generator_no_dict_no_label(all_data_per_char_as_matrix[train_mode_per_block == 1],
        #                                                 target_per_char_as_matrix[train_mode_per_block == 1], 4)
        number_of_samples_in_epoch = get_number_of_samples_per_epoch(all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0],9,10)
        # number_of_samples_in_epoch = 24000
        # temp = data_generator.next()

        # testing_data, testing_tags = get_all_triplet_combinations_testing(all_data_per_char_as_matrix,
        #                                                                   target_per_char_as_matrix,
        #                                                                   train_mode_per_block)


        # valid_data = triplet_data_collection(all_data_per_char_as_matrix[train_mode_per_block == 2],
        #                                      target_per_char_as_matrix[train_mode_per_block == 2], 80)

        total_number_of_char_in_training = 24000# all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0]/10



        from keras.models import Sequential
        from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
        from keras.layers.recurrent import LSTM, GRU
        from keras.regularizers import l2

        number_of_timestamps = 25
        _num_of_hidden_units = 100
        model = Sequential()
        model.add(LSTM(input_dim=55, output_dim=_num_of_hidden_units, input_length=number_of_timestamps,
                       return_sequences=True))
        model.add(LSTM(input_dim=_num_of_hidden_units, output_dim=_num_of_hidden_units, return_sequences=False))
        model.add(Dense(1, input_dim=1, activation='sigmoid'))


        model.compile(optimizer='rmsprop',
                      loss=identity_loss_v3)



        print "after compile"

        number_of_samples_in_epoch = 10
        model_training_results  = model.fit_generator(data_generator,number_of_samples_in_epoch, nb_epoch=1, max_q_size=1)
        final_model = model


        all_prediction_P300Net = predict_p300_model(final_model, all_data_per_char_as_matrix[train_mode_per_block != 1])

        x = T.dmatrix('x')
        import theano

        softmax_res_func = theano.function([x], T.nnet.softmax(x))

        test_tags = target_per_char_as_matrix[train_mode_per_block != 1] # np.array([target_per_char[x][train_mode_per_block != 1] for x in range(30)]).T
        all_res = test_tags

        actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1),axis=1);
        gt = np.argmax(np.mean(all_res.reshape((-1, 10, 30)), axis=1),axis=1)
        accuracy = np.sum(actual == gt) / float(len(gt))

        print "subject:{0} accu:{1}".format(subject, accuracy)
        all_model_results = dict(subject=subject,accuracy=accuracy, prediction_results=all_prediction_P300Net)

        # results = os.path.join(experiments_dir,this_file_names) + ".p"
        pickle.dump(all_model_results, file=open(os.path.join(experiments_dir,this_file_names+"_"+str(experiment_counter)+"_"+subject) + ".p", "wb"))

        # save: loss historay, final accuracy, model weight ( in order to achieve TP\FP, source data



