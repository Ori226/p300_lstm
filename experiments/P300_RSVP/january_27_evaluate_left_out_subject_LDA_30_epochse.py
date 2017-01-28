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



    # from experiments.P300_RSVP.model_left_out.load_left_out_model import  load_left_out_model

    model_file_name_full = os.path.join(os.path.dirname(LEFT_OUT_MODEL_FOLDER), model_file_name)


    # only_p300_model_1 = load_left_out_model(model_file_name)
    # original_wights = only_p300_model_1.get_weights()

    for experiment_counter, subject in enumerate(all_subjects):



        print "start subject:{}".format(subject)

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800, downsampe_params=8)

        for rep_per_sub, cross_validation_indexes in enumerate(
                list(cross_validation.KFold(len(train_mode_per_block) / 10, n_folds=4,
                                            random_state=42, shuffle=True))):
            train_data_all_subject = []
            test_data_all_subject = []

            train_tags_all_subject = []
            test_tags_all_subject = []



            def flatten_repetitions(data_to_flatten):
                return np.reshape(np.reshape(data_to_flatten.T * 10, (-1, 1)) + np.arange(10), (-1))


            train_indexes = flatten_repetitions(cross_validation_indexes[0])
            test_indexes = flatten_repetitions(cross_validation_indexes[1])

            train_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[train_indexes]).astype(np.float32))
            test_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[test_indexes]).astype(np.float32))

            train_tags_all_subject.append(target_per_char_as_matrix[train_indexes])
            test_tags_all_subject.append(target_per_char_as_matrix[test_indexes])




            from keras.optimizers import RMSprop


            print "after compile"
            from sklearn.externals import joblib
            model = joblib.load(model_file_name_full)



            train_data = stats.zscore(np.vstack(train_data_all_subject), axis=1)
            train_tags = np.vstack(train_tags_all_subject).flatten()

            test_data = stats.zscore(np.vstack(test_data_all_subject), axis=1)
            test_tags = np.vstack(test_tags_all_subject).flatten()






            accuracy_train, auc_score_train = predict_using_model(model,
                                                                  test_data.reshape(test_data.shape[0] * test_data.shape[1],
                                                                                    -1),
                                                                  test_tags)
            #
            print "{} before train accuracy_test {}:{}, auc_score_test:{} ".format(subject, rep_per_sub, accuracy_train, auc_score_train)

            accuracy_train, auc_score_train = predict_using_model(model,
                                                                  train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                                                                     -1),
                                                                  train_tags)

            print "{} before train  accuracy_train {}:{}, auc_score_train:{} ".format(subject, rep_per_sub, accuracy_train, auc_score_train)

            print "end ----------{}-------".format(file_name)



if __name__ == "__main__":

    from  experiments.P300_RSVP.model_left_out import load_left_out_model, LEFT_OUT_MODEL_FOLDER

    # load_left_out_model.load_left_out_model("RSVP_Color116msVP{}.h5".format("fat")).summary()

    # print LEFT_OUT_MODEL_FOLDER
    pass

    all_subjects_code = ['gcd', 'gcc', 'pia', 'gcb', 'gcf', 'gcg', 'gch', 'iay', 'icn', 'icr', 'fat']
    all_subjects = [(["RSVP_Color116msVP{}.mat".format(code)],
                    r"RSVP_Color116msVP{}_lda.plk".format(code)) for code in all_subjects_code]

    # all_subjects = (["RSVP_Color116msVPgcb.mat"], r"C:\git\thesis_clean_v3\experiments\P300_RSVP\model_left_out\RSVP_Color116msVPgcb.h5")

    # for i,left_out_subject in enumerate(all_subjects):
    #     training_set = list(set(all_subjects).difference(set([left_out_subject])))
    #     model_file_name = os.path.basename(left_out_subject).split(".")[0]

    for subject_data in all_subjects:
        train_on_subjset(subject_data[0], subject_data[1])
    print "stam"
    #
    #
    #
    #
    #
