import keras
from sklearn.metrics import roc_auc_score

from P300Net.data_preparation import create_data_rep_training, triplet_data_generator_no_dict, \
    get_number_of_samples_per_epoch_batch_mode, train_and_valid_generator, simple_data_generator_no_dict

from experiments.P300_RSVP.common import *
from sklearn.utils import shuffle
from keras import backend as K
from keras.models import Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras.layers import merge, Input, Dense, Flatten, Activation, Lambda, LSTM, noise, Convolution2D, \
    Permute
import scipy

__author__ = 'ORI'

from sys import platform

is_local = True
if platform != "win32":
    is_local = False

import numpy as np
import os

import argparse

rng = np.random.RandomState(42)
from os.path import basename

this_file_names = basename(__file__).split('.')[0]
print (this_file_names)


RESULTS_DIR = os.path.splitext(os.path.basename(__file__))[0]
from sklearn import cross_validation

if is_local:
    data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
    experiments_dir = r'C:\Users\ORI\Documents\Thesis\expreiment_results'

else:
    data_base_dir = r'/data_set'
    experiments_dir = r'/results'



def get_only_P300_model_LSTM_CNN(eeg_sample_shape):
    from keras.regularizers import l2
    digit_input = Input(shape=eeg_sample_shape)
    # x = Flatten(input_shape=eeg_sample_shape)(digit_input)
    from keras.layers.core import Reshape
    x = noise.GaussianNoise(sigma=0.0)(digit_input)
    x = Reshape((1, eeg_sample_shape[0], eeg_sample_shape[1]))(x)
    x = Convolution2D(nb_filter=10,
                      nb_col=eeg_sample_shape[1],
                      nb_row=1,
                      border_mode='valid',
                      init='glorot_uniform')(x)
    x = Activation('tanh')(x)
    # result shape (10,25,1)


    x = Permute((3,2, 1))(x)
    x = Reshape((eeg_sample_shape[0], 10))(x)
    # x = LSTM(10,return_sequences=True, consume_less='mem')(x)
    x = LSTM(30,return_sequences=False, consume_less='mem')(x)
    # x = LSTM(10, return_sequences=False, consume_less='mem')(x)
    # x = LSTM(100, return_sequences=False, consume_less='mem')(x)
    # x = Dense(40,activation='tanh')(x)
    x = Dense(1)(x)
    out = Activation('sigmoid')(x)


    model = Model(digit_input, out)
    model.summary()
    return model


def get_P300_model(only_P300_model, select):
    model = only_P300_model

    def identity_loss_v3(y_true, y_pred):
        y_true_reshaped = K.mean(K.reshape(y_true, (-1, select, 30)), axis=1)
        y_pred_reshaped = K.softmax(K.mean(K.reshape(y_pred, (-1, select, 30)), axis=1))
        final_val = K.mean(K.categorical_crossentropy(y_pred_reshaped, y_true_reshaped))
        return final_val + y_pred * 0

    model.compile(optimizer='adadelta',
                  loss=identity_loss_v3)
    return model

def print_predictions():
    pass


def predict_using_model(model, data, tags):
    all_prediction_P300Net = model.predict(data)
    actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
    gt = np.argmax(np.mean(tags.reshape((-1, 10, 30)), axis=1), axis=1)
    accuracy = np.sum(actual == gt) / float(len(gt))
    auc_score_test = roc_auc_score(tags.flatten(), all_prediction_P300Net)
    return accuracy, auc_score_test

np.random.seed(42)


def train_on_subjset(all_subjects, model_file_name):
    print "start ----------{}-------".format(model_file_name)
    # all_subjects = ["RSVP_Color116msVPgcd.mat",
    #                 "RSVP_Color116msVPgcc.mat",
    #                 "RSVP_Color116msVPpia.mat",
    #                 "RSVP_Color116msVPgcb.mat",
    #                 "RSVP_Color116msVPgcf.mat",
    #                 "RSVP_Color116msVPgcg.mat",
    #                 "RSVP_Color116msVPgch.mat",
    #                 # "RSVP_Color116msVPiay.mat",
    #                 "RSVP_Color116msVPicn.mat",
    #                 "RSVP_Color116msVPicr.mat",
    #                 "RSVP_Color116msVPfat.mat",
    #
    #                 ];
    #
    #
    # all_subjects = [
    #     "RSVP_Color116msVPiay.mat",
    # ];

    parser = argparse.ArgumentParser()
    parser.add_argument("-start_sub_idx", help="first sub",
                        type=int, default=0)
    parser.add_argument("-end_sub_idx", help="first sub",
                        type=int, default=len(all_subjects))
    # parser.add_argument("start_sub_idx", help="first sub",
    #                     type=int, default=len(all_subjects))
    # parser.add_argument("last_sub_idx", help="last sub",
    #                 type=int, default=len(all_subjects))

    args = parser.parse_args()
    start_idx = args.start_sub_idx
    end_idx = args.end_sub_idx

    train_data_all_subject = []
    test_data_all_subject = []

    train_tags_all_subject = []
    test_tags_all_subject = []

    for experiment_counter, subject in enumerate(all_subjects[start_idx:end_idx]):
        print "start subject:{}".format(subject)

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800, downsampe_params=8)

        for rep_per_sub, cross_validation_indexes in enumerate(
                list(cross_validation.KFold(len(train_mode_per_block) / 10, n_folds=4,
                                            random_state=42, shuffle=True))):
            batch_size = 20
            select = 1
            train_as_p300 = False
            train_indexes = train_mode_per_block == 1
            validation_indexes = train_mode_per_block == 2
            test_indexes = train_mode_per_block != 1

            if train_as_p300:

                data_generator_batch = triplet_data_generator_no_dict(all_data_per_char_as_matrix[train_indexes],
                                                                      target_per_char_as_matrix[train_indexes],
                                                                      batch_size=batch_size, select=select,
                                                                      debug_mode=False)
            else:
                # cross_validation_indexes = list(cross_validation.KFold(len(train_mode_per_block)/10, n_folds=4,
                #                                                               random_state=42, shuffle=True))

                def flatten_repetitions(data_to_flatten):
                    return np.reshape(np.reshape(data_to_flatten.T * 10, (-1, 1)) + np.arange(10), (-1))


                train_indexes = flatten_repetitions(cross_validation_indexes[0])
                test_indexes = flatten_repetitions(cross_validation_indexes[1])

                # data_generator_batch = simple_data_generator_no_dict(all_data_per_char_as_matrix[train_indexes],
                #                                                   target_per_char_as_matrix[train_indexes], shuffle_data=False)
                #
                # test_data_generator_batch = simple_data_generator_no_dict(all_data_per_char_as_matrix[train_indexes],
                #                                                      target_per_char_as_matrix[train_indexes],
                #                                                      shuffle_data=False)

                train_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[train_indexes]).astype(np.float32))
                test_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[test_indexes]).astype(np.float32))

                train_tags_all_subject.append(target_per_char_as_matrix[train_indexes])
                test_tags_all_subject.append(target_per_char_as_matrix[test_indexes])

            break
    eeg_sample_shape = (25, 55)
    only_p300_model_1 = get_only_P300_model_LSTM_CNN(eeg_sample_shape)
    only_p300_model_1.summary()

    from keras.optimizers import RMSprop

    only_p300_model_1.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'], )
    model = only_p300_model_1
    print "after compile"

    # model = LDA()
    train_data = stats.zscore(np.vstack(train_data_all_subject), axis=1)
    train_tags = np.vstack(train_tags_all_subject).flatten()

    test_data = stats.zscore(np.vstack(test_data_all_subject), axis=1)
    test_tags = np.vstack(test_tags_all_subject).flatten()

    for i in range(1):
        model.fit(train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                     train_data.shape[2], train_data.shape[3]), train_tags,
                  verbose=1, nb_epoch=30, batch_size=600, shuffle=True)

        accuracy_train, auc_score_train = predict_using_model(model,
                                                              test_data.reshape(test_data.shape[0] * test_data.shape[1],
                                                                                test_data.shape[2], test_data.shape[3]),
                                                              test_tags)

        print "accuracy_test {}:{}, auc_score_train:{} ".format(i, accuracy_train, auc_score_train)

        accuracy_train, auc_score_train = predict_using_model(model,
                                                              train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                                                                 train_data.shape[2], train_data.shape[3]),
                                                              train_tags)

        print "accuracy_train {}:{}, auc_score_train:{} ".format(i, accuracy_train, auc_score_train)

    model.optimizer.lr.set_value(0.0001)
    for i in range(1):
        model.fit(train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                     train_data.shape[2], train_data.shape[3]), train_tags,
                  verbose=1, nb_epoch=30, batch_size=600, shuffle=True)

        accuracy_train, auc_score_train = predict_using_model(model,
                                                              test_data.reshape(test_data.shape[0] * test_data.shape[1],
                                                                                test_data.shape[2], test_data.shape[3]),
                                                              test_tags)

        print "accuracy_test {}:{}, auc_score_train:{} ".format(i, accuracy_train, auc_score_train)

        accuracy_train, auc_score_train = predict_using_model(model,
                                                              train_data.reshape(
                                                                  train_data.shape[0] * train_data.shape[1],
                                                                  train_data.shape[2], train_data.shape[3]), train_tags)

        print "accuracy_train {}:{}, auc_score_train:{} ".format(i, accuracy_train, auc_score_train)
    model.save(os.path.join(experiments_dir, "model_lstm_cnn_{}.h5".format(model_file_name)), overwrite=True)
    # model.save(r"c:\temp\{}.h5".format(model_file_name,overwrite=True))
    print "end ----------{}-------".format(file_name)

    pass

    # from keras.models import Sequential
    #
    # from keras.layers import merge, Input, Dense, Flatten, Activation, Lambda, LSTM, noise
    #
    # eeg_sample_shape = (25, 55)
    # only_p300_model_1 = get_only_P300_model_LSTM(eeg_sample_shape)
    #
    # use_p300net = False
    # if use_p300net:
    #     model = get_P300_model(only_p300_model_1, select=select)
    # else:
    #
    #     only_p300_model_1.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy', metrics=['accuracy'], )
    #     model= only_p300_model_1
    # print "after compile"
    #
    #
    #
    #
    # test_tags = target_per_char_as_matrix[test_indexes]
    #
    # test_data = all_data_per_char_as_matrix[test_indexes].reshape(-1,all_data_per_char_as_matrix.shape[2],all_data_per_char_as_matrix.shape[3])
    #
    # validation_tags = target_per_char_as_matrix[validation_indexes]
    # vaidation_data = all_data_per_char_as_matrix[validation_indexes].reshape(-1,all_data_per_char_as_matrix.shape[2],all_data_per_char_as_matrix.shape[3])
    #
    #
    # train_for_inspecting_tag = target_per_char_as_matrix[train_indexes]
    # train_for_inspecting_data = all_data_per_char_as_matrix[train_indexes].reshape(-1,
    #                                                                            all_data_per_char_as_matrix.shape[2],
    #                                                                            all_data_per_char_as_matrix.shape[3])


    # np.save(os.path.join(experiments_dir, RESULTS_DIR,
    #                          subject[-7:-4] + "test_data_{}_".format(rep_per_sub) + ".npy"),test_data)
    #
    # np.save(os.path.join(experiments_dir, RESULTS_DIR,
    #                      subject[-7:-4] + "train_for_inspecting_data_{}_".format(rep_per_sub) + ".npy"), train_for_inspecting_data)
    #
    # np.save(os.path.join(experiments_dir, RESULTS_DIR,
    #                      subject[-7:-4] + "train_for_inspecting_tag_{}_".format(rep_per_sub) + ".npy"),
    #         train_for_inspecting_tag)
    #
    # np.save(os.path.join(experiments_dir, RESULTS_DIR,
    #                      subject[-7:-4] + "test_tags_{}_".format(rep_per_sub) + ".npy"),
    #         test_tags)



    # class LossHistory(keras.callbacks.Callback):
    #
    #     def on_epoch_end(self, epoch, logs={}):
    #         from sklearn.metrics import roc_auc_score
    #         if epoch  in  [0,11, 12, 13]:
    #             save_path = os.path.join(experiments_dir, RESULTS_DIR,
    #                          subject[-7:-4] + "weight_{}_{}_".format(rep_per_sub, epoch) + ".h5")
    #             self.model.save(save_path)
    #             # os.path.join( experiments_dir, RESULTS_DIR, subject[-7:-4] + "_{}_".format(rep_per_sub) + ".npy")
    #             # self.save('')
    #             all_prediction_P300Net = model.predict(stats.zscore(test_data, axis=1).astype(np.float32))
    #             actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
    #             gt = np.argmax(np.mean(test_tags.reshape((-1, 10, 30)), axis=1), axis=1)
    #             tests_accuracy = np.sum(actual == gt) / float(len(gt))
    #             auc_score_test = roc_auc_score(test_tags.flatten(), all_prediction_P300Net)
    #
    #
    #             # all_prediction_P300Net = model.predict(stats.zscore(vaidation_data, axis=1).astype(np.float32))
    #             # actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
    #             # gt = np.argmax(np.mean(validation_tags.reshape((-1, 10, 30)), axis=1), axis=1)
    #             # validation_accuracy = np.sum(actual == gt) / float(len(gt))
    #             # auc_score_validation = roc_auc_score(validation_tags.flatten(), all_prediction_P300Net)
    #
    #             all_prediction_P300Net = model.predict(stats.zscore(train_for_inspecting_data, axis=1).astype(np.float32))
    #             actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
    #             gt = np.argmax(np.mean(train_for_inspecting_tag.reshape((-1, 10, 30)), axis=1), axis=1)
    #             train_accuracy = np.sum(actual == gt) / float(len(gt))
    #             auc_score_train = roc_auc_score(train_for_inspecting_tag.flatten(), all_prediction_P300Net)
    #             from keras.callbacks import ModelCheckpoint
    #
    #             logs['tests_accuracy'] = tests_accuracy
    #             logs['accuracy_train'] = train_accuracy
    #             logs['validation_accuracy'] = train_accuracy
    #             logs['auc_score_train'] = auc_score_train
    #             logs['auc_score_test'] = auc_score_test
    #             logs['auc_score_validation'] = auc_score_test
    #             logs['subject'] = subject
    #
    #
    #
    #
    #             print "\n*{} epoch:{} mid****accuracy*: {} *accuracy_train*:{}  auc_score_train:{} auc_score_test:{} \n"\
    #                 .format(subject,epoch,tests_accuracy,train_accuracy, auc_score_train, auc_score_test)
    #
    #
    #
    # history = LossHistory()
    #
    # # model.fit_generator(data_generator_batch, 7200, 20, callbacks=[history],nb_worker=1,max_q_size=1)
    #
    # use_generator = False
    # if use_generator:
    #     log_history = model.fit_generator(data_generator_batch, 7200, 20, callbacks=[history], nb_worker=1, max_q_size=1)
    # else:
    #     samples_weight = np.ones_like(data_generator_batch[1])
    #     samples_weight[samples_weight  == 1] = 30
    #     log_history = model.fit(data_generator_batch[0], data_generator_batch[1], nb_epoch=21, batch_size=900,verbose=1,
    #                             callbacks=[history], shuffle=False, validation_split=0.1,sample_weight=samples_weight)
    #
    # results_directory =os.path.join(experiments_dir, RESULTS_DIR)
    # if not os.path.exists(results_directory):
    #     os.makedirs(results_directory)
    #
    # #np.save(os.path.join(experiments_dir, RESULTS_DIR, subject[-7:-4]+"_{}_".format(rep_per_sub)+".npy"), log_history.history)
    #
    # all_prediction_P300Net = model.predict(stats.zscore(test_data,axis=1).astype(np.float32))
    # import theano
    # import theano.tensor as T
    #
    # x = T.dmatrix('x')
    # softmax_res_func = theano.function([x], T.nnet.softmax(x))
    #
    #
    # actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
    # gt = np.argmax(np.mean(test_tags.reshape((-1, 10, 30)), axis=1), axis=1)
    # accuracy = np.sum(actual == gt) / float(len(gt))
    # print "subject:{},  accuracy: {}".format(subject, accuracy)
    # break

    # count False positive





    # print ("temp")


if __name__ == "__main__":
    all_subjects = ["RSVP_Color116msVPgcd.mat",
                    "RSVP_Color116msVPgcc.mat",
                    "RSVP_Color116msVPpia.mat",
                     "RSVP_Color116msVPgcb.mat",
                    "RSVP_Color116msVPgcf.mat",
                    "RSVP_Color116msVPgcg.mat",
                    "RSVP_Color116msVPgch.mat",
                    "RSVP_Color116msVPiay.mat",
                    "RSVP_Color116msVPicn.mat",
                    "RSVP_Color116msVPicr.mat",
                    "RSVP_Color116msVPfat.mat",
                    ];

    train_on_subjset(all_subjects, "all")

    # for left_out_subject in all_subjects[3:]:
    #     training_set = list(set(all_subjects).difference(set([left_out_subject])))
    #     model_file_name = os.path.basename(left_out_subject).split(".")[0]
    #     train_on_subjset(training_set, model_file_name)
    #     print "stam"
        




