__author__ = 'ORI'

import numpy as np
import sys
from OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4, extract_2D_channel_location

from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from scipy import stats

from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from OriKerasExtension.P300Prediction import accuracy_by_repetition, create_target_table
from keras.regularizers import l2
import pickle
import matplotlib.pyplot as plt





if __name__ == "__main__":

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
    #second_layer_model.fit(color_features, speller_train_tags, nb_epoch=1, show_accuracy=True, class_weight={0: 1, 1: 50})
    file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(data_set_locations[0])
    extract_2D_channel_location(file_name)

