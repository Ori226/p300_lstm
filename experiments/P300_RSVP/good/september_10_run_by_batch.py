from P300Net.data_preparation import create_data_rep_training, triplet_data_generator_no_dict, \
    get_number_of_samples_per_epoch_batch_mode, train_and_valid_generator

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
    # x = Dense(3)(x)
    # x = Dense(3)(x)
    out = Dense(1, activation='tanh')(x)
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

    model.compile(optimizer='SGD',
                  loss=identity_loss_v3)
    return model


def get_model_by_graph(p300_identification_model, select=3, dictionary_size=30, _eeg_sample_shape = (25, 55)):
    from keras.layers import merge, Input, Activation, Lambda
    from keras.models import Model

    def per_char_loss(X):
        concatenated = X
        reshaped = K.mean(K.reshape(concatenated, (-1, select, 30)), axis=1)

        return reshaped

    all_inputs = [Input(shape=_eeg_sample_shape) for _ in range(select * dictionary_size)]
    all_outs = [p300_identification_model(input_i) for input_i in all_inputs]

    concatenated = merge(all_outs, mode='concat')
    out = Lambda(per_char_loss, output_shape=(dictionary_size,))(concatenated)
    out = Activation('softmax')(out)

    classification_model = Model(all_inputs, out)
    classification_model.compile(optimizer='SGD',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
    return classification_model, p300_identification_model


if __name__ == "__main__":

    all_subjects = ["RSVP_Color116msVPicn.mat"];

    for experiment_counter, subject in enumerate(all_subjects):

        file_name = os.path.join(data_base_dir, subject)
        all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = create_data_rep_training(
            file_name, -200, 800, downsampe_params=8)

        batch_size = 2
        select = 3
        data_generator_batch = triplet_data_generator_no_dict(all_data_per_char_as_matrix[train_mode_per_block == 1],
                                                              target_per_char_as_matrix[train_mode_per_block == 1],
                                                              batch_size=batch_size, select=select, debug_mode=True)

        data_for_batch = data_generator_batch.next()

        data_generator_for_graph, valid_generator_all, traininig_size, validation_size = train_and_valid_generator(
            all_data_per_char_as_matrix[train_mode_per_block == 1],
            target_per_char_as_matrix[train_mode_per_block == 1], batch_size=batch_size, select=select,
            return_dict=False, debug_mode=True)

        data_for_graph = data_generator_for_graph.next()
        number_of_samples_in_epoch = get_number_of_samples_per_epoch_batch_mode(
            all_data_per_char_as_matrix[train_mode_per_block == 1].shape[0], 3, 10)

        # check the data is in the same size
        np.testing.assert_almost_equal(data_for_batch[0].reshape(batch_size, select * 30, 25, 55).transpose(1, 0, 2, 3),
                                       data_for_graph[0])

        from keras.models import Sequential

        from keras.layers import merge, Input, Dense, Flatten, Activation, Lambda


        eeg_sample_shape = (25, 55)
        only_p300_model_1 = get_only_P300_model(eeg_sample_shape)
        only_p300_model_2 = get_only_P300_model(eeg_sample_shape)
        only_p300_model_2.set_weights(only_p300_model_1.get_weights())
        model_by_graph, _ = get_model_by_graph(only_p300_model_1, select=select, _eeg_sample_shape=eeg_sample_shape)
        model = get_P300_model(only_p300_model_2, select=select)
        print "after compile"
        # for k in range(10):
        X_batch, y_batch = data_generator_batch.next()

        loss_function = model.get_loss(X_batch, y_batch)

        X_graph, y_graph = data_generator_for_graph.next()
        X_graph_list = [X_graph[i, :, :, :] for i in range(X_graph.shape[0])]
        loss_function_graph = model_by_graph.get_loss(X_graph_list, y_graph)
        np.testing.assert_almost_equal(X_batch.reshape(batch_size, select * 30, 25, 55).transpose(1, 0, 2, 3),
                                       X_graph)

        np.testing.assert_almost_equal(loss_function_graph,
                                       loss_function, decimal=6)

        new_weights_batch = model.get_weights()
        graph_new_weights_batch = model_by_graph.get_weights()
        [np.testing.assert_almost_equal(a, b) for a, b in zip(graph_new_weights_batch, new_weights_batch)]

        # in order to make sure the gradient is also the same, I'm checking the weight of the model
        for _ in range(6):
            model.train_on_batch(X_batch, y_batch)

            model_by_graph.train_on_batch(X_graph_list, y_graph)
        new_weights_batch = model.get_weights()
        graph_new_weights_batch = model_by_graph.get_weights()
        [np.testing.assert_almost_equal(a, b) for a, b in zip(model.get_weights(), model_by_graph.get_weights())]

        # just to be sure the model is really different train another instance:
        model.train_on_batch(X_batch, y_batch)

        assert not np.any(
            np.any([np.all(np.isclose(a, b)) for a, b in zip(model.get_weights(), model_by_graph.get_weights())]))

        print ("temp")
