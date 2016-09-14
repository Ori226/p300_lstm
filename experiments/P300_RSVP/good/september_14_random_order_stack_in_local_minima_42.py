import keras
import sklearn

from P300Net.data_preparation import create_data_rep_training, triplet_data_generator_no_dict, \
    get_number_of_samples_per_epoch_batch_mode, train_and_valid_generator, triplet_data_generator_no_dict_random

from experiments.P300_RSVP.common import *
from sklearn.utils import shuffle
from keras import backend as K
from keras.models import Model
import scipy

__author__ = 'ORI'

from sys import platform

is_local = True
if platform != "win32":
    is_local = False

import numpy as np
import os
from scipy import stats

rng = np.random.RandomState(42)
from os.path import basename

this_file_names = basename(__file__).split('.')[0]
print (this_file_names)



if is_local:
    data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
    experiments_dir = r'C:\Users\ORI\Documents\Thesis\expreiment_results'

else:
    data_base_dir = r'/data_set'
    experiments_dir = r'/results'


def get_only_P300_model(eeg_sample_shape):
    digit_input = Input(shape=eeg_sample_shape)
    x = Flatten(input_shape=eeg_sample_shape)(digit_input)
    x = Dense(40,activation='relu')(x)
    x = Dense(40,activation='relu')(x)
    out = Dense(1, activation='tanh')(x)
    # out = Activation('tanh')(x)


    model = Model(digit_input, out)
    return model


def get_only_P300_model_LSTM(eeg_sample_shape):
    from keras.regularizers import l2
    digit_input = Input(shape=eeg_sample_shape)
    # x = Flatten(input_shape=eeg_sample_shape)(digit_input)
    x = LSTM(100,input_shape=eeg_sample_shape,return_sequences=True, W_regularizer=l2(0.01))(digit_input)
    x = LSTM(100, return_sequences=False,W_regularizer=l2(0.01))(x)
    # x = Dense(40,activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    # out = Activation('tanh')(x)


    model = Model(digit_input, out)
    return model

def get_P300_model(only_P300_model, select):
    model = only_P300_model

    def identity_loss_v3(y_true, y_pred):
        y_true_reshaped = K.mean(K.reshape(y_true, (-1, select, 30)), axis=1)
        y_pred_reshaped = K.softmax(K.mean(K.reshape(y_pred, (-1, select, 30)), axis=1))
        final_val = K.mean(K.categorical_crossentropy(y_pred_reshaped, y_true_reshaped))
        return final_val + y_pred * 0

    model.compile(optimizer='rmsprop',
                  loss=identity_loss_v3)
    return model

def get_standard_model(only_P300_model):
    model = only_P300_model
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy')
    return model

np.random.seed(42)

if __name__ == "__main__":

    all_subjects = [
                    "RSVP_Color116msVPpia.mat",

                    "RSVP_Color116msVPgcb.mat",
                    "RSVP_Color116msVPgcc.mat",
                    "RSVP_Color116msVPgcd.mat",
                    "RSVP_Color116msVPgcf.mat",
                    "RSVP_Color116msVPgcg.mat",
                    "RSVP_Color116msVPgch.mat",
                    "RSVP_Color116msVPiay.mat",
                    "RSVP_Color116msVPicn.mat",
                    "RSVP_Color116msVPicr.mat",
                    "RSVP_Color116msVPfat.mat",
                ];

    for experiment_counter, subject in enumerate(all_subjects):

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800, downsampe_params=8)

        batch_size = 80
        select = 3
        # data_generator_batch = triplet_data_generator_no_dict(all_data_per_char_as_matrix[train_mode_per_block == 1],
        #                                                       target_per_char_as_matrix[train_mode_per_block == 1],
        #                                                       batch_size=batch_size, select=select, debug_mode=False)



        from keras.models import Sequential

        from keras.layers import merge, Input, Dense, Flatten, Activation, Lambda, LSTM

        eeg_sample_shape = (25, 55)
        only_p300_model_1 = get_only_P300_model(eeg_sample_shape)

        model = get_standard_model(only_p300_model_1)
        print "after compile"




        # in order to make sure the gradient is also the same, I'm checking the weight of the model
        # for i in range(20):
        #     train_data = data_generator_batch.next()
        #     print i

        test_tags = target_per_char_as_matrix[train_mode_per_block != 1]

        test_data = all_data_per_char_as_matrix[train_mode_per_block != 1].reshape(-1,
                                                                                   all_data_per_char_as_matrix.shape[2],
                                                                                   all_data_per_char_as_matrix.shape[3])


        train_for_inspecting_tag = target_per_char_as_matrix[train_mode_per_block == 1]
        train_for_inspecting_data = all_data_per_char_as_matrix[train_mode_per_block == 1].reshape(-1,
                                                                                   all_data_per_char_as_matrix.shape[2],
                                                                                   all_data_per_char_as_matrix.shape[3])

        class LossHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_epoch_end(self, epoch, logs={}):

                all_prediction_P300Net = model.predict(stats.zscore(test_data, axis=1).astype(np.float32))
                actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
                gt = np.argmax(np.mean(test_tags.reshape((-1, 10, 30)), axis=1), axis=1)
                accuracy_test = np.sum(actual == gt) / float(len(gt))
                auc_test = sklearn.metrics.roc_auc_score(test_tags.reshape(-1),all_prediction_P300Net.reshape(-1))


                all_prediction_P300Net = model.predict(stats.zscore(train_for_inspecting_data, axis=1).astype(np.float32))
                actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
                gt = np.argmax(np.mean(train_for_inspecting_tag.reshape((-1, 10, 30)), axis=1), axis=1)
                accuracy_train = np.sum(actual == gt) / float(len(gt))

                auc_train = sklearn.metrics.roc_auc_score(train_for_inspecting_tag.reshape(-1), all_prediction_P300Net.reshape(-1))

                logs['accuracy_train'] = accuracy_train
                logs['auc_train'] = auc_train

                logs['accuracy_test'] = accuracy_test
                logs['auc_test'] = auc_test

                self.losses.append(logs)
                print "\n*mid****accuracy*: {} *accuracy_train*:{} \n".format(accuracy_test,accuracy_train )



        history = LossHistory()

        train_data = stats.zscore(all_data_per_char_as_matrix[train_mode_per_block == 1],axis=2)
        train_data = train_data.reshape(-1,train_data.shape[2],train_data.shape[3])
        train_tags = target_per_char_as_matrix[train_mode_per_block == 1].reshape(-1)

        shuffle_train_data, shuffled_tags = shuffle(train_data, train_tags, random_state=42)
        sample_weigt = np.ones_like(shuffled_tags)
        sample_weigt[shuffled_tags] = 1
        model.fit(shuffle_train_data,shuffled_tags,
                  batch_size=240, nb_epoch=50, callbacks=[history], shuffle=True,sample_weight=sample_weigt)

        temp = history.losses

        import matplotlib.pyplot as plt
        import pandas as pd
        pd.DataFrame(temp).plot()
        plt.show()
        all_prediction_P300Net = model.predict(stats.zscore(test_data,axis=1).astype(np.float32))

        print ("stam")

        actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
        gt = np.argmax(np.mean(test_tags.reshape((-1, 10, 30)), axis=1), axis=1)
        accuracy = np.sum(actual == gt) / float(len(gt))
        print "accuracy: {}".format(accuracy)







        print ("temp")
