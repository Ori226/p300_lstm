from experiments.P300_RSVP.common import *


__author__ = 'ORI'

import numpy as np
import os
import cPickle as pickle
# I should learn how to load libraries in a more elegant way



from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
# from OriKerasExtension import ThesisHelper

# reload(OriKerasExtension)
# reload(OriKerasExtension.ThesisHelper)
from   OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
import OriKerasExtension.P300Prediction

reload(OriKerasExtension.P300Prediction)
from OriKerasExtension.P300Prediction import accuracy_by_repetition, create_target_table

# import OriKerasExtension
#
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from keras.layers.recurrent import LSTM
# from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

# reload(OriKerasExtension)


rng = np.random.RandomState(42)


def create_evaluation_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1, testing_mode=1):
    #     gcd_res = readCompleteMatFile(file_name)
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'], gcd_res['target'],
                                    fist_time_stamp, last_time_stamp)
    # print  data_for_eval

    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    test_data_gcd, test_target_gcd = temp_data_for_eval[gcd_res['train_mode'] != testing_mode], data_for_eval[1][
        gcd_res['train_mode'] != testing_mode]
    return test_data_gcd, test_target_gcd


def downsample_data(data, number_of_original_samples, down_samples_param):
    new_number_of_time_stamps = number_of_original_samples / down_samples_param

    # print  data_for_eval
    temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))

    for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
        temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
    return temp_data_for_eval


def create_training_and_testing(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                                take_same_number_positive_and_negative=False):
    train_data, train_tags = create_train_data(gcd_res, fist_time_stamp, last_time_stamp, down_samples_param,
                                               take_same_number_positive_and_negative)
    test_data, test_tags = create_evaluation_data(gcd_res=gcd_res, fist_time_stamp=fist_time_stamp,
                                                  last_time_stamp=last_time_stamp,
                                                  down_samples_param=down_samples_param)
    func_args = dict(fist_time_stamp=fist_time_stamp, last_time_stamp=last_time_stamp,
                     down_samples_param=down_samples_param,
                     take_same_number_positive_and_negative=take_same_number_positive_and_negative)
    return train_data, train_tags, test_data, test_tags, func_args

def create_only_testing(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                                take_same_number_positive_and_negative=False):



    test_data, test_tags = create_evaluation_data(gcd_res=gcd_res, fist_time_stamp=fist_time_stamp,
                                                  last_time_stamp=last_time_stamp,
                                                  down_samples_param=down_samples_param, testing_mode=3)
    func_args = dict(fist_time_stamp=fist_time_stamp, last_time_stamp=last_time_stamp,
                     down_samples_param=down_samples_param,
                     take_same_number_positive_and_negative=take_same_number_positive_and_negative)
    return None, None, test_data, test_tags, func_args

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
#     all_tags = gcd_res['target'][gcd_res['train_mode'] == 1]
#
#
#
#
#     all_data = temp_data_for_eval[gcd_res['train_mode'] == 1]
#
#     categorical_tags = to_categorical(all_tags)
#
#     # shuffeled_samples, suffule_tags = shuffle(all_data, all_tags, random_state=0)
#
#
#     return all_data, all_tags

def create_train_data_from_all(all_gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                      take_same_number_positive_and_negative=False):


    all_positive_train = []
    all_negative_train = []
    all_data =None
    all_tags = None
    for i, single_subject_data in enumerate(all_gcd_res):
        data_for_eval = ExtractDataVer4(single_subject_data['all_relevant_channels'], single_subject_data['marker_positions'],
                                        single_subject_data['target'], fist_time_stamp, last_time_stamp)

        temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)
        current_subject_tag = single_subject_data['target'][single_subject_data['train_mode'] != 1]
        current_subject_data = temp_data_for_eval[single_subject_data['train_mode'] != 1]
        if i == 0:
            all_tags = current_subject_tag
            all_data = current_subject_data
        else:
            all_tags = np.hstack([all_tags,current_subject_tag])
            all_data = np.vstack([all_data,current_subject_data])



        # shuffeled_samples, suffule_tags = shuffle(all_data, all_tags, random_state=0)


    return all_data, all_tags


def create_data_for_compare_by_repetition(file_name):
    gcd_res = readCompleteMatFile(file_name)
    sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
                       train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
                       stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])
    return sub_gcd_res


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
# class EvaluateByRepetition(object):
#     def __init__(self, subject_file):
#         super(EvaluateByRepetition, self).__init__()
#         self.sub_gcd_res = create_data_for_compare_by_repetition(file_name)
#
#     def foo(self, actual, prediction):
#         _, _, gt_data_for_sum = create_target_table(self.sub_gcd_res, actual)
#         _, _, actual_data_for_sum = create_target_table(self.sub_gcd_res, prediction[:, 1])
#
#         all_accuracies = dict([
#                                   [rep, accuracy_by_repetition(actual_data_for_sum, gt_data_for_sum,
#                                                                number_of_repetition=rep)]
#                                   for rep in range(10)])
#
#         print ", ".join([
#                             "acc {}:{}".format(k, v)
#                             for k, v in all_accuracies.iteritems()])
#         return all_accuracies


class LSTM_CNN_EEG_Comb(GeneralModel):
    def get_params(self):
        super(LSTM_CNN_EEG_Comb, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_CNN_EEG_Comb, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_CNN_EEG_Comb, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_CNN_EEG_Comb, self).__init__()
        self.positive_weight = positive_weight
        self._num_of_hidden_units = _num_of_hidden_units

        '''
        define the neural network model:

        '''
        # from keras.layers.extra import *

        from keras.models import Sequential
        # from keras.initializations import norRemal, identity
        from keras.layers.recurrent import GRU, LSTM
        from keras.layers.convolutional import Convolution2D
        from keras.layers.core import Dense, Activation, TimeDistributedDense, Reshape
        # from keras.layers.wrappers import TimeDistributed
        from keras.layers.convolutional import MaxPooling2D
        from keras.layers.core import Permute

        from keras.regularizers import l2, activity_l2


        maxToAdd = 200
        # define our time-distributed setup
        model = Sequential()

        model.add(Reshape((1, maxToAdd, 55), input_shape=(maxToAdd, 55)))  # this line updated to work with keras 1.0.2
        # model.add(TimeDistributedDense(10, input_shape=(maxToAdd, 55)))
        model.add(Convolution2D(3, 12, 55, border_mode='valid', W_regularizer=l2(0.1)))  # org
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(12, 1), border_mode='valid'))
        model.add(Permute((2, 1, 3)))
        model.add(Reshape((model.layers[-1].output_shape[1],
                           model.layers[-1].output_shape[2])))  # this line updated to work with keras 1.0.2
        model.add(LSTM(output_dim=10, return_sequences=False))
        #
        model.add(Dense(2, activation='softmax'))


        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy')
        self.model = model

        # model.predict(np.random.rand(28, 200, 55).astype(np.float32)).shape

        print model.layers[-1].output_shape
        # print "2 {} {}".format(model.layers[1].output_shape[-3:], (1, maxToAdd, np.prod(model.layers[1].output_shape[-3:])))
        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def fit(self, _X, y):
        from keras.callbacks import ModelCheckpoint

        _y = to_categorical(y)
        # _X = np.expand_dims(np.expand_dims(_X,3),4).transpose([0,1,3,2,4])


        # (batch, times, color_channel, x, y)

        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)
        sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))

        self.model.fit(stats.zscore(_X, axis=1), _y,
                       nb_epoch=1, show_accuracy=True, verbose=1,
                       class_weight={0: 1, 1: self.positive_weight})

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))

if __name__ == "__main__":
    model_20 = None
    model_100 = None

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

    all_subjects = ["RSVP_Color116msVPicr.mat","RSVP_Color116msVPpia.mat"]

    data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
    # model = LDA()

    all_models = [LSTM_CNN_EEG_Comb(50, 20)]
    for model_type in all_models:

        # all_model_results = [create_training_and_testing(gcd_res, -200, 800, 1, False)]
        all_model_results = []
        all_res = [readCompleteMatFile(os.path.join(data_base_dir, subject)) for subject in all_subjects]
        training_data, train_tags,  = create_train_data_from_all(all_res, -200, 800, 1, False)
        model = model_type
        model.fit(training_data, train_tags)

        for subject in all_subjects:
            file_name = os.path.join(data_base_dir, subject)
            gcd_res = readCompleteMatFile(file_name)
            repetition_eval = EvaluateByRepetition(file_name)

            for data_extraction_method in [create_only_testing(gcd_res, -200, 800, 1, False)
                                           ]:

                # create_training_and_testing(gcd_res, 0, 400, 1, True)
                _, _, testing_data, test_tags, func_args = data_extraction_method
                # type: GeneralModel
                print "starting {}:{}:{}".format(subject, model.get_name()[-7:-4], ",".join([str(x) for x in func_args.values()]))

                # training_data = create_train_data(gcd_res, 0, 400, 1, True)
                # testing_data = create_evaluation_data(gcd_res, 0, 400, 1)

                # training_data, train_tags, testing_data, test_tags = create_training_and_testing(gcd_res, 0, 400, 1, True)

                # model = My_LDA()
                # model.fit(training_data, train_tags)
                prediction_res = model.predict(testing_data)
                all_accuracies = repetition_eval.foo(test_tags, prediction_res)
                all_model_results.append(
                    dict(all_accuracies=all_accuracies, subject_name=subject, model=model.get_name(),
                         model_params=model.get_params(), func_args=func_args))
                # model.reset()
                print "end {}:{}:{}".format(subject, model.get_name()[-7:-4],
                                                 ",".join([str(x) for x in func_args.values()]))




        pickle.dump(all_model_results, file=open(model_type.get_name() + "_all_b.p", "wb"))


    pass
