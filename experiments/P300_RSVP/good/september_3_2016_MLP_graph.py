from keras.layers import GRU

from P300Net.data_preparation import create_data_rep_training, triplet_data_generator, triplet_data_collection, \
    triplet_data_generator_no_dict, get_number_of_samples_per_epoch, train_and_valid_generator
from P300Net.models import get_item_subgraph, get_graph,get_graph_lstm,  get_item_lstm_subgraph
from experiments.P300_RSVP.common import *
from sklearn.utils import shuffle
import theano.tensor as T
import scipy
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




def get_model_by_graph(select=3,dictionary_size=30):
    from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, Lambda
    from keras.models import Model
    from keras.layers.recurrent import LSTM

    # first, define the vision modules
    from keras import backend as K

    def per_char_loss(X):
        # alls = X.values()
        concatenated = X  # K.concatenate(alls)
        reshaped = K.mean(K.reshape(concatenated, (-1,select, 30)), axis=1)

        return reshaped  # K.softmax(K.reshape(reshaped, (reshaped.shape[0], -1)))

    eeg_sample_shape = (25,55)
    # eeg_sample_shape = (200, 55)
    digit_input = Input(shape=eeg_sample_shape)

    if False:
        x = Flatten(input_shape=eeg_sample_shape)(digit_input)
        x = Dense(2)(x)
        x = Activation('tanh')(x)
    # # x = Dropout(0.1)(x)
    # # x = Dense(40)(x)
    # # x = Dropout(0.1)(x)
    # # x = Activation('tanh')(x)
    # x = Dense(1)(x)
    else:
        x = LSTM(100, return_sequences=True)(digit_input)
        x = LSTM(100, return_sequences=False)(x)
        x = Activation('tanh')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(40)(x)
    # x = Dropout(0.1)(x)
    # x = Activation('tanh')(x)
    x = Dense(1)(x)
    out = x  # Flatten()(x)



    # P300 identification model
    p300_identification_model = Model(digit_input, out)

    all_inputs = [Input(shape=eeg_sample_shape) for _ in range(select*dictionary_size)]

    all_outs = [p300_identification_model(input_i) for input_i in all_inputs]

    concatenated = merge(all_outs, mode='concat')
    out = Lambda(per_char_loss, output_shape=(dictionary_size,))(concatenated)
    out = Activation('softmax')(out)
    # out = Dense(30)(concatenated)


    # out = Dense(1, activation='sigmoid')(concatenated)

    classification_model = Model(all_inputs, out)
    return classification_model, p300_identification_model


data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'



if __name__ == "__main__":

    all_subjects = ["RSVP_Color116msVPpia.mat",
                    "RSVP_Color116msVPicr.mat",


                    "RSVP_Color116msVPgcb.mat",
                    "RSVP_Color116msVPgcc.mat",
                    "RSVP_Color116msVPgcd.mat",
                    "RSVP_Color116msVPgcf.mat",
                    "RSVP_Color116msVPgcg.mat",
                    "RSVP_Color116msVPgch.mat",
                    "RSVP_Color116msVPiay.mat",
                    "RSVP_Color116msVPicn.mat"
                    "RSVP_Color116msVPfat.mat",
                    ];

    for subject in all_subjects:
        # subject = "RSVP_Color116msVPgcd.mat"

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800,downsampe_params=8)

        # data_generator = triplet_data_generator(all_data_per_char_as_matrix[train_mode_per_block == 1], target_per_char_as_matrix[train_mode_per_block == 1], batch_size=80,select=3,return_dict=False)
        # data_generator = triplet_data_generator(all_data_per_char_as_matrix[train_mode_per_block == 1],
        #                                         target_per_char_as_matrix[train_mode_per_block == 1], batch_size=80,
        #                                         select=3, return_dict=False)

        data_generator, valid_generator_all, traininig_size, validation_size = train_and_valid_generator(all_data_per_char_as_matrix[train_mode_per_block == 1], target_per_char_as_matrix[train_mode_per_block == 1], batch_size=80,select=3,return_dict=False)

        from itertools import  tee
        (valid_generator, valid_generator_for_debugging) = tee(valid_generator_all,2)


        # data_generator = triplet_data_generator_no_dict(all_data_per_char_as_matrix[train_mode_per_block == 1], target_per_char_as_matrix[train_mode_per_block == 1], 80)
        number_of_samples_in_epoch = get_number_of_samples_per_epoch(all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0],3,10)
        # temp = data_generator.next()

        # testing_data, testing_tags = get_all_triplet_combinations_testing(all_data_per_char_as_matrix,
        #                                                                   target_per_char_as_matrix,
        #                                                                   train_mode_per_block)



        # region Build the P300Net model
        # model = get_graph_lstm(3, 10, 25,55)
        # endregion


        #
        #
        # # region the P300Net identification model
        # P300IdentificationModel = get_item_lstm_subgraph(25, 55)
        # print "before compile"
        # P300IdentificationModel.compile(loss='binary_crossentropy', class_mode="binary", optimizer='rmsprop')
        # print "after compile"
        # endregion

        # region train the P300Net model
        # model.fit_generator(data_generator, 2880, nb_epoch=10, validation_data=valid_data)
        # model.fit_generator(data_generator, 80*5, nb_epoch=20, validation_data=valid_data)
        # endregion

        # number_of_samples_in_epoch = 30*3*80
        # temp = next(data_generator)
        (model,P300IdentificationModel) = get_model_by_graph(select=3)
        model.compile(optimizer='rmsprop',
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])

        import keras


        def evaluate_accuracy(predictions, tags):
            actual = np.argmax(np.mean(predictions.reshape((-1, 10, 30)), axis=1), axis=1);
            gt = np.argmax(np.mean(tags.reshape((-1, 10, 30)), axis=1), axis=1)
            accuracy = np.sum(actual == gt) / float(len(gt))
            return accuracy

        class LossHistory(keras.callbacks.Callback):
            def on_train_end(self, logs={}):
                print "!!!temp!!!"
                all_prediction_P300Net = predict_p300_model(P300IdentificationModel,
                                                            all_data_per_char_as_matrix[train_mode_per_block != 1])

                test_tags = target_per_char_as_matrix[train_mode_per_block != 1]  # np.array([target_per_char[x][train_mode_per_block != 1] for x in range(30)]).T
                self.model.predict()

                valid_generator_for_debugging
                print "{} \n".format(evaluate_accuracy(all_prediction_P300Net, test_tags))
                self.losses = []

            def on_batch_end(self, batch, logs={}):

                all_prediction_P300Net = predict_p300_model(P300IdentificationModel,
                                                            all_data_per_char_as_matrix[train_mode_per_block != 1])

                test_tags = target_per_char_as_matrix[
                    train_mode_per_block != 1]  # np.array([target_per_char[x][train_mode_per_block != 1] for x in range(30)]).T
                res = model.evaluate_generator(valid_generator_for_debugging,validation_size)
                print "evalute_generatr {0} \n".format(res)
                # print evaluate_accuracy(all_prediction_P300Net, test_tags)
                self.losses = []
                self.losses.append(logs.get('loss'))




        history = LossHistory()
        loss_hisotry = model.fit_generator(data_generator,2880, nb_epoch=5, nb_worker=1,max_q_size=1)
        # loss_hisotry = model.fit_generator(data_generator, traininig_size, nb_epoch=20, validation_data=valid_generator,
        #                                    nb_val_samples=validation_size)
        print "temp"

        # all_train_data = dict()
        # train_p300_model(P300IdentificationModel, all_data_per_char_as_matrix[train_mode_per_block == 1],
        #                  target_per_char_as_matrix[train_mode_per_block == 1])



        # final_model = get_item_lstm_subgraph(25,55)
        # final_model_original_weights = final_model.get_weights()

        # final_model.compile(loss='binary_crossentropy', class_mode="binary", optimizer='sgd')
        # final_model.set_weights(list(model.nodes['item_latent'].layer.get_weights()))



        all_prediction_P300Net = predict_p300_model(P300IdentificationModel, all_data_per_char_as_matrix[train_mode_per_block != 1])
        # all_prediction_normal = predict_p300_model(P300IdentificationModel, all_data_per_char_as_matrix[train_mode_per_block != 1])
        # all_prediction_normal = all_prediction_P300Net
        plt.subplot(1, 4, 1)
        # plt.imshow(all_prediction, interpolation='none')
        plt.subplot(1, 4, 2)
        x = T.dmatrix('x')
        import theano

        softmax_res_func = theano.function([x], T.nnet.softmax(x))

        #
        # plt.imshow(softmax_res_func(all_prediction), interpolation='none')
        # plt.subplot(1, 4, 3)
        # plt.imshow(softmax_res_func(np.mean(all_prediction.reshape((-1, 10, 30)), axis=1)).astype(np.int),
        #            interpolation='none')

        plt.subplot(1, 4, 3)
        test_tags = target_per_char_as_matrix[train_mode_per_block != 1] # np.array([target_per_char[x][train_mode_per_block != 1] for x in range(30)]).T
        # plt.imshow(np.mean(all_res.reshape((-1, 10, 30)), axis=1), interpolation='none')




        all_res = test_tags

        # plt.imshow(softmax_res_func(all_prediction_normal.reshape((-1, 30))), interpolation='none')



        # actual_untrained = np.argmax(softmax_res_func(np.mean(all_prediction_normal.reshape((-1, 10, 30)),axis=1)),axis=1)
        actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1),axis=1);
        gt = np.argmax(np.mean(all_res.reshape((-1, 10, 30)), axis=1),axis=1)
        # np.intersect1d(actual, gt)
        # accuracy = 0 #np.sum(actual == gt) / float(len(gt))
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(softmax_res_func(np.mean(all_prediction_normal.reshape((-1, 10, 30)), axis=1)))
        # plt.subplot(1, 2, 2)
        # # plt.imshow(np.mean(all_res.reshape((-1, 10, 30)), axis=1))
        # plt.show()

        # accuracy = 0
        accuracy = np.sum(actual == gt) / float(len(gt))
        print "subject:{0} accu:{1} ".format(subject, accuracy)
        # target_per_char_as_matrix[0:10,:]


