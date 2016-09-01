from P300Net.data_preparation import create_data_rep_training, triplet_data_generator, triplet_data_collection
from P300Net.models import get_item_subgraph, get_graph
from experiments.P300_RSVP.common import *

__author__ = 'ORI'

import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.RandomState(42)



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
    all_data = np.zeros((magic_number, number_in_batch, 200, 55), dtype=np.float32)
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
              sample_weight=np.ones((shuffle_flatten_tags.shape[0])) + shuffle_flatten_tags.flatten() * 30)
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

    for subject in all_subjects:
        # subject = "RSVP_Color116msVPgcd.mat"

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800)

        data_generator = triplet_data_generator(all_data_per_char_as_matrix[train_mode_per_block == 1], target_per_char_as_matrix[train_mode_per_block == 1], 80)


        valid_data = triplet_data_collection(all_data_per_char_as_matrix[train_mode_per_block == 2],
                                             target_per_char_as_matrix[train_mode_per_block == 2], 80)
        # temp = next(data_generator)

        # res = get_all_triplet_combinations_indexes_only(all_data_per_char_as_matrix,
        #                                                 target_per_char_as_matrix)

        temp = create_train_test_validation(all_data_per_char_as_matrix, target_per_char_as_matrix)

        # all_data = create_data_for_compare_by_repetition(file_name)

        # get_all_triplet_combinations(all_data_per_char, target_per_char)
        # extract % of the data for validation. it should contains several 10 repetition

        total_number_of_char_in_training = all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0]/10




        # res = get_all_triplet_combinations_indexes_only(all_data_per_char_as_matrix,
        #                                                                   target_per_char_as_matrix)
        # import sklearn


        testing_data, testing_tags = get_all_triplet_combinations_testing(all_data_per_char_as_matrix,
                                                                          target_per_char_as_matrix,
                                                                          train_mode_per_block)

        #
        # validation_data, validation_tags = get_all_triplet_combinations(
        #     all_data_per_char_as_matrix[train_mode_per_block == 2],
        #     target_per_char_as_matrix[train_mode_per_block == 2])

        # model = get_graph(3, 10)
        temp_model = get_item_subgraph(None, None)


        # validation_input_dict = dict(
        #     [["positive_item_input_{}".format(i), stats.zscore(validation_data[i], axis=1)] for i in
        #      range(90)])
        #
        # validation_input_dict['triplet_loss'] = validation_tags

        temp_model.compile(loss='binary_crossentropy', class_mode="binary", optimizer='sgd')

        # model.fit_generator(data_generator, len(all_data_per_char_as_matrix), nb_epoch = 1)
        # model.fit_generator(data_generator, 2880, nb_epoch=5, validation_data=valid_data)

        all_train_data = dict()
        train_p300_model(temp_model, all_data_per_char_as_matrix[train_mode_per_block == 1],
                         target_per_char_as_matrix[train_mode_per_block == 1])
        #
        # for x in range(30):
        #     all_train_data[x] = all_data_per_char[x][train_mode_per_block == 1]
        #
        all_test_data = dict()
        for x in range(30):
            all_test_data[x] = all_data_per_char[x][train_mode_per_block != 1]

        # predict_p300_model(temp_model, all_train_data)

        # for epoch in range(2):
        #     print "starting {0}".format(epoch)
        #     training_data, train_tags = get_all_triplet_combinations(
        #         all_data_per_char_as_matrix[train_mode_per_block == 1],
        #         target_per_char_as_matrix[train_mode_per_block == 1],
        #         train_mode_per_block)
        #     input_dict = dict(
        #         [["positive_item_input_{}".format(i), stats.zscore(training_data[i], axis=1)] for i in
        #          range(90)])
        #
        #     input_dict['triplet_loss'] = train_tags
        #     model.train_on_batch(input_dict)
        final_model = get_item_subgraph(None, None)
        final_model_original_weights = final_model.get_weights()
        # temp_weight = list(model.nodes['item_latent'].layer.get_weights())
        final_model.compile(loss='binary_crossentropy', class_mode="binary", optimizer='sgd')
        # final_model.set_weights(temp_weight)
        import theano.tensor as T

        all_prediction = np.zeros((all_data_per_char[0][train_mode_per_block != 1].shape[0], 30))
        for stimuli_i in range(30):
            all_prediction[:, stimuli_i] = final_model.predict(
                stats.zscore(all_data_per_char[stimuli_i][train_mode_per_block != 1], axis=1)).flatten()
            # input_dict = dict([["positive_item_input_{}".format(i),np.random.rand(1,55,200).astype(np.float32)] for i in range(60)])

        all_prediction_untrained = predict_p300_model(temp_model, all_test_data)
        # plt.subplot(1, 4, 1)
        # plt.imshow(all_prediction, interpolation='none')
        # plt.subplot(1, 4, 2)
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


