__author__ = 'ORI'
from experiments.P300_RSVP.common import *
from matplotlib.collections import LineCollection
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.io
import sys, os
import cPickle as pickle
import matplotlib.pyplot as plt
# I should learn how to load libraries in a more elegant way




import OriKerasExtension
# from OriKerasExtension import ThesisHelper

# reload(OriKerasExtension)
# reload(OriKerasExtension.ThesisHelper)
from   OriKerasExtension.ThesisHelper import LoadSingleSubjectPython, readCompleteMatFile, ExtractDataVer4
import OriKerasExtension.P300Prediction

reload(OriKerasExtension.P300Prediction)
from OriKerasExtension.P300Prediction import accuracy_by_repetition, create_target_table

# import OriKerasExtension
#
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from keras.layers.recurrent import LSTM
# from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit

# reload(OriKerasExtension)


from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

rng = np.random.RandomState(42)


def create_evaluation_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1):
    #     gcd_res = readCompleteMatFile(file_name)
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'], gcd_res['target'],
                                    fist_time_stamp, last_time_stamp)
    # print  data_for_eval

    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    test_data_gcd, test_target_gcd = temp_data_for_eval[gcd_res['train_mode'] != 1], data_for_eval[1][
        gcd_res['train_mode'] != 1]
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
    train_data, train_tags = create_train_data_split(gcd_res, fist_time_stamp, last_time_stamp, down_samples_param,
                                               take_same_number_positive_and_negative)
    test_data, test_tags = create_evaluation_data(gcd_res=gcd_res, fist_time_stamp=fist_time_stamp,
                                                  last_time_stamp=last_time_stamp,
                                                  down_samples_param=down_samples_param)
    func_args = dict(fist_time_stamp=fist_time_stamp, last_time_stamp=last_time_stamp,
                     down_samples_param=down_samples_param,
                     take_same_number_positive_and_negative=take_same_number_positive_and_negative)
    return train_data, train_tags, test_data, test_tags, func_args


def create_train_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                      take_same_number_positive_and_negative=False):
    all_positive_train = []
    all_negative_train = []

    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    # # extract the calibration_data
    # positive_train_data_gcd = temp_data_for_eval[
    #     np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 1], axis=0)]
    # negative_train_data_gcd = temp_data_for_eval[
    #     np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 0], axis=0)]

    all_tags = gcd_res['target'][gcd_res['train_mode'] == 1]


    # all_positive_train.append(positive_train_data_gcd)
    # all_negative_train.append(negative_train_data_gcd)

    all_data = temp_data_for_eval[gcd_res['train_mode'] == 1]
    # if take_same_number_positive_and_negative:
    #     negative_train_data_gcd = rng.permutation(np.vstack(all_negative_train))[0:positive_train_data_gcd.shape[0]]
    # else:
    #     negative_train_data_gcd = np.vstack(all_negative_train)
    #
    # all_data = np.vstack([positive_train_data_gcd, negative_train_data_gcd])

    # all_tags = np.vstack(
    #     [np.ones((positive_train_data_gcd.shape[0], 1)), np.zeros((negative_train_data_gcd.shape[0], 1))]).astype(np.int)
    categorical_tags = to_categorical(all_tags)

    # shuffeled_samples, suffule_tags = shuffle(all_data, all_tags, random_state=0)
    

    return all_data, all_tags

def create_train_data_split(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                      take_same_number_positive_and_negative=False):


    all_positive_train = []
    all_negative_train = []

    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    # extract the calibration_data
    positive_train_data_gcd = temp_data_for_eval[
        np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 1], axis=0)]
    negative_train_data_gcd = temp_data_for_eval[
        np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 0], axis=0)]

    negative_splits = np.split(shuffle(negative_train_data_gcd), 29)
    all_data_splits = []
    all_tag_splits = []
    for i in range(29):
        suffuled_split_data, suffuled_split_tag = shuffle(np.vstack([positive_train_data_gcd, negative_splits[i]]),
                                                          np.vstack([np.ones((positive_train_data_gcd.shape[0], 1)),
                                                                     np.zeros((negative_splits[i].shape[0], 1))]).astype(
                                                              np.int))
        all_data_splits.append(suffuled_split_data)
        all_tag_splits.append(suffuled_split_tag)

    return all_data_splits, all_tag_splits


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
#
#
# class GeneralModel(object):
#     __metaclass__ = ABCMeta
#
#     @abstractmethod
#     def predict(self, _X):
#         pass
#
#     @abstractmethod
#     def fit(self, _X, _y):
#         pass
#
#     @abstractmethod
#     def reset(self):
#         pass
#
#     @abstractmethod
#     def get_name(self):
#         pass
#
#     @abstractmethod
#     def get_params(self):
#         pass


# class My_LDA(LDA, GeneralModel):
#     def reset(self):
#         super(My_LDA, self).reset()
#
#     def predict(self, _X):
#         return super(My_LDA, self).predict_proba(_X.reshape(_X.shape[0], -1))
#
#     def fit(self, _X, _y):
#         return super(My_LDA, self).fit(_X.reshape(_X.shape[0], -1), _y.flatten(), store_covariance=None, tol=None)
#
#     def get_name(self):
#         super(My_LDA, self).get_name()
#         return self.__class__.__name__
#
#     def get_params(self):
#         return None
#
# class My_LDA_norm(LDA, GeneralModel):
#     def reset(self):
#         super(My_LDA_norm, self).reset()
#
#     def predict(self, _X):
#         _X = stats.zscore(_X, axis=1)
#         return super(My_LDA_norm, self).predict_proba(_X.reshape(_X.shape[0], -1))
#
#     def fit(self, _X, _y):
#         _X =stats.zscore(_X, axis=1)
#         return super(My_LDA_norm, self).fit(_X.reshape(_X.shape[0], -1), _y.flatten(), store_covariance=None, tol=None)
#
#     def get_name(self):
#         super(My_LDA_norm, self).get_name()
#         return self.__class__.__name__
#
#     def get_params(self):
#         return None


class LSTM_EEG(GeneralModel):
    def get_params(self):
        super(LSTM_EEG, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_EEG, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_EEG, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_EEG, self).__init__()
        self.positive_weight = positive_weight
        self._num_of_hidden_units = _num_of_hidden_units

        '''
        define the neural network model:

        '''
        from keras.models import Sequential
        from keras.layers.recurrent import LSTM
        from keras.layers.core import Dense, Dropout, Activation
        from keras.regularizers import l2

        self.model = Sequential()
        self.model.add(LSTM(input_dim=55, output_dim=_num_of_hidden_units, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(input_dim=_num_of_hidden_units, output_dim=_num_of_hidden_units, return_sequences=False))
        self.model.add(Dense(2, W_regularizer=l2(0.06)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def fit(self, _X, y):
        from keras.callbacks import ModelCheckpoint

        _y = to_categorical(y)

        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)
        sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))
        self.model.fit(stats.zscore(_X[sss[0][0]], axis=1), _y[sss[0][0]],
                       nb_epoch=20, show_accuracy=True, verbose=1, validation_data=(
                stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
                       class_weight={0: 1, 1: self.positive_weight},
                       callbacks=[checkpointer])

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))


class LSTM_CNN_EEGEnsamble(GeneralModel):
    def get_params(self):
        super(LSTM_CNN_EEGEnsamble, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_CNN_EEGEnsamble, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_CNN_EEGEnsamble, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_CNN_EEGEnsamble, self).__init__()
        self.positive_weight = positive_weight
        self._num_of_hidden_units = _num_of_hidden_units
        self.number_of_subset = 29
        '''
        define the neural network model:

        '''


        from keras.models import Sequential
        from keras.layers.recurrent import SimpleRNN, LSTM, GRU
        from keras.layers.convolutional import Convolution2D, Convolution1D
        from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten
        from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
        from keras.layers.core import Permute

        maxToAdd = 200
        model = Sequential()

        model.add(TimeDistributedDense(10, input_shape=(maxToAdd, 55)))
        model.add(Activation('tanh'))
        model.add(
            Reshape((1, maxToAdd, 10)))  # this line updated to work with keras 1.0.2
        model.add(Convolution2D(3, 20, 1, border_mode='valid')) # org
        model.add(Activation('tanh'))
        model.add(Convolution2D(1, 1, 1, border_mode='same'))  # org
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(20, 1), border_mode='valid'))
        model.add(Permute((2, 1, 3)))
        model.add(Reshape((9, 10)))  # this line updated to work with keras 1.0.2
        model.add(LSTM(output_dim=50, return_sequences=False))
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
        self.all_weight = dict([(i, self.model.get_weights())for i in range(self.number_of_subset)])

    def fit(self, _X, y):
        counter = 0
        for X_sub_db, y_sub_db in zip(_X,y):
            print "start fitting {}".format(counter)
            _y = to_categorical(y_sub_db)
            self.model.set_weights(self.all_weight[counter])
            self.model.fit(stats.zscore(X_sub_db, axis=1), _y,
                           nb_epoch=100, batch_size=128, show_accuracy=True, verbose=1)
            self.all_weight[counter] = self.model.get_weights()
            counter += 1

    def predict(self, _X):

        all_prediction_results = np.zeros((self.number_of_subset, _X.shape[0],2))
        for counter in range(self.number_of_subset):
            self.model.set_weights(self.all_weight[counter])
            all_prediction_results[counter,:,:] = self.model.predict(stats.zscore(_X, axis=1))

        return np.mean(all_prediction_results, axis=0)



class LSTM_CNN_EEG(GeneralModel):
    def get_params(self):
        super(LSTM_CNN_EEG, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_CNN_EEG, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_CNN_EEG, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_CNN_EEG, self).__init__()
        self.positive_weight = positive_weight
        self._num_of_hidden_units = _num_of_hidden_units

        '''
        define the neural network model:

        '''
        # from keras.layers.extra import *
        import numpy as  np
        from keras.models import Sequential
        from keras.layers.recurrent import LSTM
        from keras.layers.core import Dense, Dropout, Activation
        from keras.regularizers import l2

        from keras.datasets import mnist
        from keras.models import Sequential
        # from keras.initializations import norRemal, identity
        from keras.layers.recurrent import SimpleRNN, LSTM, GRU
        from keras.optimizers import RMSprop, Adadelta
        from keras.layers.convolutional import Convolution2D, Convolution1D
        from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten
        # from keras.layers.wrappers import TimeDistributed
        from keras.models import model_from_json
        from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
        from keras.layers.core import Permute



        size = 28
        maxToAdd = 200
        # define our time-distributed setup
        model = Sequential()

        model.add(TimeDistributedDense(10, input_shape=(maxToAdd, 55)))
        # model.add(Convolution2D(1, 1, 10, border_mode='valid', input_shape=(1,maxToAdd, 55)))
        model.add(Activation('tanh'))
        model.add(
            Reshape((1, maxToAdd, 10)))  # this line updated to work with keras 1.0.2
        model.add(Convolution2D(3, 20, 1, border_mode='valid')) # org
        model.add(Activation('tanh'))
        model.add(Convolution2D(1, 1, 1, border_mode='same'))  # org
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(20, 1), border_mode='valid'))
        model.add(Permute((2, 1, 3)))
        model.add(Reshape((9, 10)))  # this line updated to work with keras 1.0.2
        model.add(GRU(output_dim=20, return_sequences=False))
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
        # self.model.fit(stats.zscore(_X[sss[0][0]], axis=1), _y[sss[0][0]],
        #                nb_epoch=50, show_accuracy=True, verbose=1, validation_data=(
        #         stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
        #                class_weight={0: 1, 1: self.positive_weight},
        #                callbacks=[checkpointer])

        self.model.fit(stats.zscore(_X, axis=1), _y,
                       nb_epoch=50, show_accuracy=True, verbose=1,
                       class_weight={0: 1, 1: self.positive_weight})

        # self.model.fit(_X[sss[0][0]], _y[sss[0][0]],
        #                nb_epoch=20)

        # self.model.predict(_X[sss[0][0]], batch_size=2)

    def predict(self, _X):
        # _X = np.expand_dims(np.expand_dims(_X, 3), 4).transpose([0, 1, 3, 2, 4])

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

    # all_subjects = ["RSVP_Color116msVPicr.mat"]

    data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
    # model = LDA()

    all_models = [LSTM_CNN_EEGEnsamble(30,20)]
    for model_type in all_models:

        all_model_results = []

        for subject in all_subjects:
            file_name = os.path.join(data_base_dir, subject)
            gcd_res = readCompleteMatFile(file_name)
            repetition_eval = EvaluateByRepetition(file_name)

            for data_extraction_method in [create_training_and_testing(gcd_res, -200, 800, 1, False)
                                           ]:

                # create_training_and_testing(gcd_res, 0, 400, 1, True)
                training_data, train_tags, testing_data, test_tags, func_args = data_extraction_method
                model = model_type  # type: GeneralModel
                print "starting {}:{}:{}".format(subject, model.get_name()[-7:-4], ",".join([str(x) for x in func_args.values()]))

                # training_data = create_train_data(gcd_res, 0, 400, 1, True)
                # testing_data = create_evaluation_data(gcd_res, 0, 400, 1)

                # training_data, train_tags, testing_data, test_tags = create_training_and_testing(gcd_res, 0, 400, 1, True)

                # model = My_LDA()
                model.fit(training_data, train_tags)
                prediction_res = model.predict(testing_data)
                all_accuracies = repetition_eval.foo(test_tags, prediction_res)
                all_model_results.append(
                    dict(all_accuracies=all_accuracies, subject_name=subject, model=model.get_name(),
                         model_params=model.get_params(), func_args=func_args))
                model.reset()
                print "end {}:{}:{}".format(subject, model.get_name()[-7:-4],
                                                 ",".join([str(x) for x in func_args.values()]))

            break




        pickle.dump(all_model_results, file=open(model_type.get_name() + "_b_for_new_loss.p", "wb"))


    pass
