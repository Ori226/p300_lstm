from abc import ABCMeta, abstractmethod

import numpy as np
from keras.utils.np_utils import to_categorical
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from OriKerasExtension.P300Prediction import create_target_table, accuracy_by_repetition
from OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
# from experiments.P300_RSVP.july_2_LSTM_CNN_Comb import create_data_for_compare_by_repetition, file_name

# from experiments.P300_RSVP.july_8_MLP import create_train_data





def create_train_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                      take_same_number_positive_and_negative=False):
    all_positive_train = []
    all_negative_train = []

    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    all_tags = gcd_res['target'][gcd_res['train_mode'] == 1]




    all_data = temp_data_for_eval[gcd_res['train_mode'] == 1]

    categorical_tags = to_categorical(all_tags)

    # shuffeled_samples, suffule_tags = shuffle(all_data, all_tags, random_state=0)


    return all_data, all_tags

class GeneralModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, _X):
        pass

    @abstractmethod
    def fit(self, _X, _y):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


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
        self.model.add(LSTM(input_shape=(200,55), output_dim=_num_of_hidden_units, input_length=25, return_sequences=False))
        self.model.add(Dropout(0.3))
        # self.model.add(LSTM(input_dim=_num_of_hidden_units, output_dim=_num_of_hidden_units, return_sequences=False))
        self.model.add(Dense(2, W_regularizer=l2(0.06)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def fit(self, _X, y):


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

        from keras.models import Sequential
        # from keras.initializations import norRemal, identity
        from keras.layers.recurrent import GRU
        from keras.layers.convolutional import Convolution2D
        from keras.layers.core import Dense, Activation, TimeDistributedDense, Reshape
        # from keras.layers.wrappers import TimeDistributed
        from keras.layers.convolutional import MaxPooling2D
        from keras.layers.core import Permute




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

        self.model.fit(stats.zscore(_X, axis=1), _y,
                       nb_epoch=50, show_accuracy=True, verbose=1,
                       class_weight={0: 1, 1: self.positive_weight})

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))


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
        self.model.fit(stats.zscore(_X[sss[0][0]], axis=1), _y[sss[0][0]],
                       nb_epoch=20, show_accuracy=True, verbose=1, validation_data=(
                stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
                       class_weight={0: 1, 1: self.positive_weight},
                       callbacks=[checkpointer])

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))





class LSTM_CNN_EEG_Comb_new_loss(GeneralModel):
    def get_params(self):
        super(LSTM_CNN_EEG_Comb_new_loss, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_CNN_EEG_Comb_new_loss, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_CNN_EEG_Comb_new_loss, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_CNN_EEG_Comb_new_loss, self).__init__()
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
        self.model.fit(stats.zscore(_X[sss[0][0]], axis=1), _y[sss[0][0]],
                       nb_epoch=20, show_accuracy=True, verbose=1, validation_data=(
                stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
                       class_weight={0: 1, 1: self.positive_weight},
                       callbacks=[checkpointer])

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))

class CNN_2011_EEG(GeneralModel):
    def __init__(self, positive_weight, original_fs=200):
        super(CNN_2011_EEG, self).__init__()

        from keras.models import Sequential
        from keras.layers.convolutional import Convolution2D
        from keras.layers.core import Dense, Activation, Flatten, Reshape
        from keras.layers.convolutional import MaxPooling2D
        from keras.regularizers import l2
        number_of_time_stamps = original_fs
        number_of_in_channels = 55
        number_of_out_channels = 10
        self.model = Sequential()
        self.positive_weight = positive_weight
        self.model.add(Reshape((1, number_of_time_stamps, number_of_in_channels), input_shape=(number_of_time_stamps, number_of_in_channels)))
        self.model.add(Convolution2D(nb_filter=10,
                                nb_col=number_of_out_channels,
                                nb_row=1,
                                input_shape=(1, number_of_time_stamps, number_of_in_channels),
                                border_mode='same',
                                init='glorot_normal',W_regularizer=l2(0.01)))

        self.model.add(Activation('tanh'))
        self.model.add(MaxPooling2D(pool_size=(1, number_of_in_channels)))
        self.model.add(
            Convolution2D(nb_filter=number_of_out_channels, nb_row=6, nb_col=1, border_mode='same', init='glorot_normal',W_regularizer=l2(0.01)))
        self.model.add(MaxPooling2D(pool_size=(20, 1)))
        self.model.add(Activation('tanh'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def predict(self, _X):
        super(CNN_2011_EEG, self).predict(_X)
        return self.model.predict(stats.zscore(_X, axis=1))

    def get_params(self):
        return self.model.get_weights()

    def fit(self, _X, _y):
        _y = to_categorical(_y)
        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)
        sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))


        self.model.fit(stats.zscore(_X[sss[0][0]], axis=1), _y[sss[0][0]],
                       nb_epoch=30, show_accuracy=True, verbose=1,
                       validation_data=(
                           stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
                       class_weight={0: 1, 1: self.positive_weight},
                       callbacks=[checkpointer] )

    def get_name(self):
        return self.__class__.__name__ + "_" + str(self.positive_weight)


    def reset(self):
        self.model.set_weights(self.original_weights)


def P300_Speller_liss(y_true, y_pred):
    from keras import backend as K
    return K.mean(y_pred - 0 * y_true)


class CNN_MLP(GeneralModel):
    def __init__(self, positive_weight):
        super(CNN_MLP, self).__init__()
        from keras.models import Sequential
        from keras.layers.convolutional import Convolution2D
        from keras.layers.core import Dense, Activation, Flatten, Reshape
        from keras.layers.convolutional import MaxPooling2D
        from keras.regularizers import l2
        number_of_time_stamps = 200
        number_of_in_channels = 55
        number_of_out_channels =10
        self.model = Sequential()
        self.positive_weight = positive_weight
        self.model.add(Flatten(input_shape=(number_of_time_stamps, number_of_in_channels)))
        self.model.add(Dense(40)) #, input_shape=(number_of_time_stamps, number_of_in_channels)))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(40))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))

        if True:
            self.model.compile(loss=P300_Speller_liss, optimizer='rmsprop')
        else:
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def predict(self, _X):
        super(CNN_MLP, self).predict(_X)
        return self.model.predict(stats.zscore(_X, axis=1))

    def get_params(self):
        return self.model.get_weights()

    def fit(self, _X, _y):
        _y = to_categorical(_y)
        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)

        sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))

        validation_data = (
                              stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
        class_weight = {0: 1, 1: self.positive_weight},
        callbacks = [checkpointer]


        self.model.fit(stats.zscore(_X, axis=1), _y,
                       nb_epoch=5, show_accuracy=True, verbose=1,
                       class_weight={0: 1, 1: self.positive_weight})

    def get_name(self):
        return self.__class__.__name__ + "_" + str(self.positive_weight)


    def reset(self):
        self.model.set_weights(self.original_weights)


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


def create_training_and_testing(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
                                take_same_number_positive_and_negative=False):
    train_data, train_tags = create_train_data(gcd_res, fist_time_stamp, last_time_stamp, down_samples_param)
    test_data, test_tags = create_evaluation_data(gcd_res=gcd_res, fist_time_stamp=fist_time_stamp,
                                                  last_time_stamp=last_time_stamp,
                                                  down_samples_param=down_samples_param)
    func_args = dict(fist_time_stamp=fist_time_stamp, last_time_stamp=last_time_stamp,
                     down_samples_param=down_samples_param,
                     take_same_number_positive_and_negative=take_same_number_positive_and_negative)
    return train_data, train_tags, test_data, test_tags, func_args


class EvaluateByRepetition(object):
    def __init__(self, file_name):
        super(EvaluateByRepetition, self).__init__()
        self.sub_gcd_res = create_data_for_compare_by_repetition(file_name)

    def foo(self, actual, prediction):
        _, _, gt_data_for_sum = create_target_table(self.sub_gcd_res, actual)
        _, _, actual_data_for_sum = create_target_table(self.sub_gcd_res, prediction[:, 1])

        all_accuracies = dict([
                                  [rep, accuracy_by_repetition(actual_data_for_sum, gt_data_for_sum,
                                                               number_of_repetition=rep)]
                                  for rep in range(1,11)])

        print ", ".join([
                            "acc {}:{}".format(k, v)
                            for k, v in all_accuracies.iteritems()])
        return all_accuracies


def downsample_data(data, number_of_original_samples, down_samples_param):
    new_number_of_time_stamps = number_of_original_samples / down_samples_param

    # print  data_for_eval
    temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))

    for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
        temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
    return temp_data_for_eval


def create_data_for_compare_by_repetition(file_name):
    gcd_res = readCompleteMatFile(file_name)
    sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
                       train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
                       stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])
    return sub_gcd_res


def create_evaluation_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1):
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'], gcd_res['target'],
                                    fist_time_stamp, last_time_stamp)

    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    test_data_gcd, test_target_gcd = temp_data_for_eval[gcd_res['train_mode'] != 1], data_for_eval[1][
        gcd_res['train_mode'] != 1]
    return test_data_gcd, test_target_gcd