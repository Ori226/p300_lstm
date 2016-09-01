
# coding: utf-8

# In[2]:

from matplotlib.collections import LineCollection


import matplotlib.pyplot as plt

import numpy as np 
import scipy.io
import sys,os
import matplotlib.pyplot as plt
# I should learn how to load libraries in a more elegant way
# sys.path.append(r'C:\Users\ori22_000\Documents\IDC-non-sync\Thesis\PythonApplication1\OriKerasExtension')
data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'

#sys.path.append(r'C:\git\thesis_clean_v2\OriKerasExtension')
import OriKerasExtension

from OriKerasExtension.ThesisHelper import *
import cPickle as pickle






# In[28]:

def LoadSingleSubjectPythonWithTime(file_name, start, end):
    res = readCompleteMatFile(file_name);

    all_data, all_tags = ExtractDataVer2(res['all_relevant_channels'], res['marker_positions'], res['target'], start, end);

    trasposed_data = all_data.transpose(0, 2, 1)

    trasposed_data = trasposed_data.reshape(trasposed_data.shape[0], -1)

    all_target = trasposed_data[np.where(all_tags == 1)[0], :]
    all_non_target = trasposed_data[np.where(all_tags != 1)[0], :]

    subset_size = all_target.shape[0]
    np.random.seed(0)

    all_target = np.random.permutation(all_target)[0:subset_size,
                 :]  # FromFileListToArray(target_files, 'all_target_flatten', 600)
    all_non_target = np.random.permutation(all_non_target)[0:subset_size,
                     :]  # FromFileListToArray(non_target_files, 'all_non_target_flatten',600 )

    return [all_target, all_non_target]



'''
suffle the samples in order to balance between the target and non target on training:
'''






# In[3]:

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from  keras.regularizers import WeightRegularizer

script_file_name = os.path.splitext(os.path.basename(__file__))[0]




if __name__ == "__main__":
    '''
    define the neural network model:
    '''
    model = Sequential()

    model.add(LSTM(input_dim=55, output_dim=20, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(input_dim=20, output_dim=20, return_sequences=False))
    model.add(Dense(1, W_regularizer=l2(0.06)))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')

    # save the original weight before start the learning
    original_weights = model.get_weights();

    # In[ ]:

    from sklearn import datasets, linear_model, cross_validation, grid_search
    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split

    from keras.callbacks import ModelCheckpoint

    number_of_channels = 55

    data_set_locations = ["RSVP_Color116msVPicr.mat",
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

    all_results = []
    all_models = []

    for subject_file_name in data_set_locations:
        [all_target, all_non_target] = LoadSingleSubjectPythonWithTime(os.path.join(data_base_dir, subject_file_name),
                                                                       0, 400)
        all_samples = np.vstack((all_target, all_non_target))
        all_tags = np.vstack((np.ones((all_target.shape[0], 1)), np.zeros((all_non_target.shape[0], 1))))

        shuffeled_samples, suffule_tags = shuffle(all_samples, all_tags, random_state=0)

        kf_total = cross_validation.KFold(shuffeled_samples.shape[0], n_folds=4, shuffle=True, random_state=4)
        k_fold_iterator = 0
        for train, test in kf_total:
            # initalize the weiget to the pre-training ones

            data_traing_lstm = shuffeled_samples[train].reshape(shuffeled_samples[train].shape[0], number_of_channels,
                                                                -1).transpose(0, 2, 1)
            print (shuffeled_samples[test].shape)
            train_tags = suffule_tags[train]

            data_testing_lstm = shuffeled_samples[test].reshape(shuffeled_samples[test].shape[0], number_of_channels,
                                                                -1).transpose(0, 2, 1)
            test_tags = suffule_tags[test]

            model.set_weights(original_weights)
            checkpointer = ModelCheckpoint(filepath=r"c:\temp\weights.hdf5", verbose=1, save_best_only=True)
            model.fit(data_traing_lstm, train_tags, nb_epoch=40, show_accuracy=True, verbose=0,
                      callbacks=[checkpointer], validation_split=0.1)
            all_models.append(dict(
                dict(subject_name=subject_file_name,
                     k_fold_iterator=k_fold_iterator,
                     model_weight=model.get_weights())))

            evaluation_results = model.evaluate(data_testing_lstm, test_tags, show_accuracy=True, verbose=0)
            print (evaluation_results)
            all_results.append(dict(subject_name=subject_file_name, k_fold_iterator=k_fold_iterator,
                                    evaluation_results=evaluation_results))
            print "{}:{}:{}".format(subject_file_name, k_fold_iterator,evaluation_results )
            k_fold_iterator += 1


    pickle.dump(all_results, open("save_" + script_file_name + ".p", "wb"))
    pickle.dump(all_models, open("save_model_" + script_file_name + ".p", "wb"))
