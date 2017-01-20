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
    x = Permute((3,2, 1))(x)
    x = Reshape((eeg_sample_shape[0], 10))(x)
    x = LSTM(30,return_sequences=False, consume_less='mem')(x)
    x = Dense(1)(x)
    out = Activation(activation='sigmoid')(x)


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



    for cross_validation_iter in range(4):


        train_data_all_subject = []
        test_data_all_subject = []

        train_tags_all_subject = []
        test_tags_all_subject = []
        test_data_all_subject_with_noise = dict()
        for experiment_counter, subject in enumerate(all_subjects[start_idx:end_idx]):
            print "start subject:{}".format(subject)


            file_name = os.path.join(data_base_dir, subject)
            _, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
                file_name, -200, 800, downsampe_params=8)


            noise_data = dict()

            noist_shifts = [-120, -80, -40, 0, 40, 80, 120]

            for time_shift_noise in noist_shifts:
                _, _, _, noise_data[time_shift_noise], _ = create_data_rep_training(
                    file_name, (-200 + time_shift_noise), (800 + time_shift_noise), downsampe_params=8)




            for rep_per_sub, cross_validation_indexes in enumerate(list(cross_validation.KFold(len(train_mode_per_block)/10, n_folds=4,
                                                                                  random_state=42, shuffle=True))):
                if cross_validation_indexes < cross_validation_iter:
                    continue


                batch_size = 20
                select = 1
                train_as_p300 = False
                train_indexes = train_mode_per_block == 1
                validation_indexes = train_mode_per_block == 2
                test_indexes = train_mode_per_block != 1


                # cross_validation_indexes = list(cross_validation.KFold(len(train_mode_per_block)/10, n_folds=4,
                #                                                               random_state=42, shuffle=True))

                def flatten_repetitions(data_to_flatten):
                    return np.reshape(np.reshape(data_to_flatten.T * 10, (-1, 1)) + np.arange(10), (-1))

                train_indexes = flatten_repetitions(cross_validation_indexes[0])
                test_indexes = flatten_repetitions(cross_validation_indexes[1])

                train_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[train_indexes]).astype(np.float32))
                test_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[test_indexes]).astype(np.float32))

                for time_shift_noise in noist_shifts:
                    if time_shift_noise not in test_data_all_subject_with_noise:
                        test_data_all_subject_with_noise[time_shift_noise] = []
                    test_data_all_subject_with_noise[time_shift_noise].append(np.asarray(noise_data[time_shift_noise][test_indexes]).astype(np.float32))


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
        test_data_with_noise = dict()
        for time_shift_noise in noist_shifts:
            test_data_with_noise[time_shift_noise] = stats.zscore(np.vstack(test_data_all_subject_with_noise[time_shift_noise]), axis=1)
        # test_data = stats.zscore(np.vstack(test_data_all_subject), axis=1)
        test_tags = np.vstack(test_tags_all_subject).flatten()


        for i in  range(30):
            model.fit(train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                         train_data.shape[2], train_data.shape[3]), train_tags,
                      verbose=1,nb_epoch=1, batch_size=600,shuffle=True)



            for time_shift_noise in noist_shifts:
                test_data = test_data_with_noise[time_shift_noise]
                accuracy_test, auc_score_test = predict_using_model(model,
                                                                    test_data.reshape(
                                                                        test_data.shape[0] * test_data.shape[1],
                                                                        test_data.shape[2], test_data.shape[3]), test_tags)
                print "cv:{} noise:{} accuracy_test {}:{}, auc_score_train:{} ".format(cross_validation_iter, time_shift_noise, i, accuracy_test, auc_score_test )



            accuracy_train, auc_score_train = predict_using_model(model,
                                                                  train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                                                                     train_data.shape[2], train_data.shape[3]), train_tags)

            print "cv:{} accuracy_train {}:{}, auc_score_train:{} ".format(cross_validation_iter, i, accuracy_train, auc_score_train)

        model.optimizer.lr.set_value(0.0001)
        for i in range(6):
            model.fit(train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                         train_data.shape[2], train_data.shape[3]), train_tags,
                      verbose=1, nb_epoch=5, batch_size=600, shuffle=True)

            for time_shift_noise in noist_shifts:
                test_data = test_data_with_noise[time_shift_noise]
                accuracy_test, auc_score_test = predict_using_model(model,
                                                                    test_data_with_noise[time_shift_noise].reshape(
                                                                        test_data.shape[0] * test_data.shape[1],
                                                                        test_data.shape[2], test_data.shape[3]), test_tags)
                print "cv:{} noise:{} accuracy_test {}:{}, auc_score_train:{} ".format(cross_validation_iter, time_shift_noise, i, accuracy_test,
                                                                                 auc_score_test)

            accuracy_train, auc_score_train = predict_using_model(model,
                                                                  train_data.reshape(
                                                                      train_data.shape[0] * train_data.shape[1],
                                                                      train_data.shape[2], train_data.shape[3]), train_tags)

            print "cv:{} accuracy_train {}:{}, auc_score_train:{} ".format(cross_validation_iter, i, accuracy_train, auc_score_train)

    pass









