# coding: utf-8

# In[13]:

__author__ = 'ORI'

from matplotlib.collections import LineCollection

import numpy as np
import scipy.io
import sys, os
import matplotlib.pyplot as plt
# I should learn how to load libraries in a more elegant way




from OriKerasExtension import ThesisHelper
from OriKerasExtension.ThesisHelper import LoadSingleSubjectPython, readCompleteMatFile, ExtractDataVer4
from OriKerasExtension.P300Prediction import accuracy_by_repetition, create_target_table
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit


reload(ThesisHelper)

from sklearn.utils import shuffle


# [all_target, all_non_target] = LoadSingleSubjectPython(r'C:\Users\ORI\Documents\Thesis\dataset_all\RSVP_Color116msVPfat.mat')






# all_samples = np.vstack((all_target,all_non_target))


# '''
# Create the tagging column
# '''
# all_tags = np.vstack((np.ones((all_target.shape[0],1)), np.zeros((all_non_target.shape[0],1))))



# from OriKerasExtension.OriKerasExtension import DebugLSTM
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

'''
define the neural network model:
'''


def create_compile_cnn_model():
    model = Sequential()

    number_of_time_stamps = 20
    number_of_out_channels = 10
    number_of_in_channels = 55
    #length_of_time_axe_mask = 10

    model.add(Convolution2D(nb_filter=10,
                            nb_col=number_of_out_channels,
                            nb_row=1,
                            input_shape=(1, number_of_time_stamps, number_of_in_channels),
                            border_mode='same',
                            init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, number_of_in_channels)))
    model.add(
        Convolution2D(nb_filter=number_of_out_channels, nb_row=6, nb_col=1, border_mode='same', init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(20, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def create_compile_lstm_model():
    '''
    define the neural network model:
    '''
    model_lstm = Sequential()

    model_lstm.add(LSTM(input_dim=55, output_dim=20, return_sequences=False))
    model_lstm.add(Dense(2))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm


def create_compile_dense_model():
    '''
    define the neural network model:
    '''
    model_lstm = Sequential()
    model_lstm.add(keras.layers.core.Flatten(input_shape=(55, 20)))
    model_lstm.add(Dense(input_dim=55 * 20, output_dim=20))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(2))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm


# def down_sample_data()


def create_evaluation_data(file_name, down_samples_param):
    gcd_res = readCompleteMatFile(file_name)
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'], gcd_res['target'],
                                    -200, 800)
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


def create_train_data(file_name, down_samples_param):
    all_positive_train = []
    all_negative_train = []

    others = ["RSVP_Color116msVPgcd.mat"]

    for other_file_name in others:
        file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(other_file_name)
        gcd_res = readCompleteMatFile(file_name)
        last_time_stamp = 800
        fist_time_stamp = -200

        data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                        gcd_res['target'], fist_time_stamp, last_time_stamp)

        # total_time = last_time_stamp - fist_time_stamp
        # number_of_original_samples = total_time / 5
        # new_number_of_time_stamps = number_of_original_samples / down_samples_param
        #
        #
        # # print  data_for_eval
        # temp_data_for_eval = np.zeros((data_for_eval[0].shape[0], new_number_of_time_stamps, data_for_eval[0].shape[2]))
        #
        # for new_i, i in enumerate(range(0, 200, new_number_of_time_stamps)):
        #     temp_data_for_eval[:, new_i, :] = np.mean(data_for_eval[0][:, range(i, (i + new_number_of_time_stamps)), :], axis=1)
        print data_for_eval[0].shape
        temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

        positive_train_data_gcd = temp_data_for_eval[
            np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 1], axis=0)]
        negative_train_data_gcd = temp_data_for_eval[
            np.all([gcd_res['train_mode'] == 1, gcd_res['target'] == 0], axis=0)]
        all_positive_train.append(positive_train_data_gcd)
        all_negative_train.append(negative_train_data_gcd)

    positive_train_data_gcd = np.vstack(all_positive_train)
    negative_train_data_gcd = np.vstack(all_negative_train)

    all_data = np.vstack([positive_train_data_gcd, negative_train_data_gcd])

    all_tags = np.vstack(
        [np.ones((positive_train_data_gcd.shape[0], 1)), np.zeros((negative_train_data_gcd.shape[0], 1))])
    categorical_tags = to_categorical(all_tags)

    shuffeled_samples, suffule_tags = shuffle(all_data, categorical_tags, random_state=0)
    # shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags


def create_data_for_compare_by_repetition(file_name):
    gcd_res = readCompleteMatFile(file_name)
    sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
                       train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
                       stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])
    return sub_gcd_res


# shuffeled_samples, suffule_tags = create_train_data(file_name=None, down_samples_param=5)
# shuffeled_samples, suffule_tags = create_train_data(file_name=None, down_samples_param=20)
model = create_compile_lstm_model()
original_weights = model.get_weights()

# data_set_locations = ["RSVP_Color116msVPicr.mat",
#                       "RSVP_Color116msVPpia.mat",
#                       "RSVP_Color116msVPfat.mat",
#                       "RSVP_Color116msVPgcb.mat",
#                       "RSVP_Color116msVPgcc.mat",
#                       "RSVP_Color116msVPgcd.mat",
#                       "RSVP_Color116msVPgcf.mat",
#                       "RSVP_Color116msVPgcg.mat",
#                       "RSVP_Color116msVPgch.mat",
#                       "RSVP_Color116msVPiay.mat",
#                       "RSVP_Color116msVPicn.mat"];

data_set_locations = ["RSVP_Color116msVPgcd.mat"]

results = []

for subject_name in data_set_locations:
    file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(subject_name)
    down_sample_param = 5
    shuffeled_samples, suffule_tags = create_train_data(file_name=file_name, down_samples_param=down_sample_param)
    print shuffeled_samples.shape

    for i in range(1):
        model.set_weights(original_weights)

        sss = list(StratifiedShuffleSplit(suffule_tags[:, 0], n_iter=1, test_size=0.1))
        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds.hdf5", verbose=1, save_best_only=True)
        print shuffeled_samples[sss[0][0]].shape
        print stats.zscore(shuffeled_samples[sss[0][0]], axis=1).shape
        model.fit(stats.zscore(shuffeled_samples[sss[0][0]], axis=1), suffule_tags[sss[0][0]],
                  nb_epoch=20, show_accuracy=True, verbose=1, validation_data=(
                stats.zscore(shuffeled_samples[sss[0][1]], axis=1), suffule_tags[sss[0][1]]),
                  class_weight={0: 1, 1: 50},
                  callbacks=[checkpointer])

        test_data_gcd, test_target_gcd = create_evaluation_data(file_name=file_name,
                                                                down_samples_param=down_sample_param)

        test_prediction = model.predict(stats.zscore(test_data_gcd, axis=1), verbose=1)

        x, y, _ = roc_curve(test_target_gcd, test_prediction[:, 1])



        # This is the ROC curve
        # plt.plot(x, y)
        # plt.show()
        auc_score = roc_auc_score(test_target_gcd, test_prediction[:, 1])
        print "auc_score:{0}".format(auc_score)
        sub_gcd_res = create_data_for_compare_by_repetition(file_name)
        # sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
        # train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
        # stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])

        _, _, gt_data_for_sum = create_target_table(sub_gcd_res, test_target_gcd)
        _, _, actual_data_for_sum = create_target_table(sub_gcd_res, test_prediction[:, 1])

        print "accuracy_by_repetition {0}".format(
            accuracy_by_repetition(actual_data_for_sum, gt_data_for_sum, number_of_repetition=10))

    results.append(dict(subject_name=subject_name, test_prediction=test_prediction, auc_score=auc_score))
    break;


# In[4]:

import keras


# In[20]:




# In[73]:

import theano

result_func = theano.function([model.get_input(train=False)], model.layers[-1].get_output(train=False)[:, 0])
# convolutions = convout1_f(reshaped[img_to_visualize: img_to_visualize+1])


# In[77]:

# result_func.grad
from theano import tensor as T

grad_func = T.grad(model.layers[-1].get_output(train=False)[0, 0], model.get_input(train=False))


# In[110]:

result_func(stats.zscore(shuffeled_samples[sss[0][0]], axis=1).astype('float32')[0].reshape(1, 40, 55)).shape
# print stats.zscore(shuffeled_samples[sss[0][0]], axis=1).astype('float32')[0].reshape(1,40,55).shape

dlogistic = theano.function([model.get_input(train=False)], grad_func)


data_to_diagnots = stats.zscore(test_data_gcd, axis=1).astype('float32')[38].reshape(1, 40, 55)

temp = dlogistic(data_to_diagnots)
temp.shape
plt.imshow(temp[0, :, :].T, interpolation='none')
plt.show()
plt.imshow(data_to_diagnots[0, :, :].T, interpolation='none')
plt.show()

data_to_diagnots = stats.zscore(test_data_gcd, axis=1).astype('float32')[39].reshape(1, 40, 55)

temp = dlogistic(data_to_diagnots)
temp.shape
plt.imshow(temp[0, :, :].T, interpolation='none')
plt.show()
plt.imshow(data_to_diagnots[0, :, :].T, interpolation='none')
plt.show()

data_to_diagnots = stats.zscore(test_data_gcd, axis=1).astype('float32')[40].reshape(1, 40, 55)

temp = dlogistic(data_to_diagnots)
temp.shape
plt.imshow(temp[0, :, :].T, interpolation='none')
plt.show()
plt.imshow(data_to_diagnots[0, :, :].T, interpolation='none')
plt.show()

data_to_diagnots = stats.zscore(test_data_gcd, axis=1).astype('float32')[74].reshape(1, 40, 55)

temp = dlogistic(data_to_diagnots)
temp.shape
plt.imshow(temp[0, :, :].T, interpolation='none')
plt.show()
plt.imshow(data_to_diagnots[0, :, :].T, interpolation='none')
plt.show()

data_to_diagnots = stats.zscore(test_data_gcd, axis=1).astype('float32')[371].reshape(1, 40, 55)

temp = dlogistic(data_to_diagnots)
temp.shape
plt.imshow(temp[0, :, :].T, interpolation='none')
plt.show()
plt.imshow(data_to_diagnots[0, :, :].T, interpolation='none')
plt.show()

data_to_diagnots = stats.zscore(test_data_gcd, axis=1).astype('float32')[264].reshape(1, 40, 55)

temp = dlogistic(data_to_diagnots)
temp.shape
plt.imshow(temp[0, :, :].T, interpolation='none')
plt.show()
plt.imshow(data_to_diagnots[0, :, :].T, interpolation='none')
plt.show()

data_to_diagnots = stats.zscore(test_data_gcd, axis=1).astype('float32')[265].reshape(1, 40, 55)

temp = dlogistic(data_to_diagnots)
temp.shape
plt.imshow(temp[0, :, :].T, interpolation='none')
plt.show()
plt.imshow(data_to_diagnots[0, :, :].T, interpolation='none')
plt.show()



# In[41]:

model.layers[-1].get_output(train=False)[0]

# >>> x = T.dmatrix('x')
# >>> s = T.sum(1 / (1 + T.exp(-x)))
# >>> gs = T.grad(s, x)


# In[43]:

result_func[0, :]


# In[57]:


x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = theano.function([x, y], z[:, 0])


# In[58]:

f(np.asarray([[1, 2]]), np.asarray([[1, 2]]))


# In[66]:

T.grad(z[:, 0], x)


# In[127]:

np.where(test_prediction[:, 1] > 0.12)


# In[106]:

np.where(test_target_gcd == 1)


# In[131]:

test_data_gcd.shape, test_target_gcd.shape


# In[132]:

# result_func = theano.function([model.get_input(train=False)], model.layers[-1].get_output(train=False)[:,0])
model.layers


# In[133]:

precalssify_layer = theano.function([model.get_input(train=False)], model.layers[0].get_output(train=False))


# In[141]:

middle_layer_test_data = precalssify_layer(stats.zscore(test_data_gcd, axis=1).astype('float32'))


# In[168]:

import matplotlib.cm as cm

for i in range(100):
    plt.subplot(1, 2, 1)
    plt.imshow(np.hstack([middle_layer_test_data[i * 30:(i + 1) * 30, :],
                          test_target_gcd.reshape(test_target_gcd.shape[0], 1)[i * 30:(i + 1) * 30]]),
               interpolation='none')
    plt.subplot(1, 2, 2)
    #     print test_target_gcd.reshape(test_target_gcd.shape[0],1)
    plt.imshow(test_target_gcd.reshape(test_target_gcd.shape[0], 1)[i * 30:(i + 1) * 30], cmap=cm.Greys_r)
    plt.show()
# plt.subplot(1,2,1)
#     plt.imshow(middle_layer_test_data[30:60,:],interpolation='none')
#     plt.subplot(1,2,2)
#     plt.imshow(test_target_gcd.reshape(test_target_gcd.shape[0],1)[30:60], cmap = cm.Greys_r)
#     plt.show()
#     reshaped_middle_layer_test_data = middle_layer_test_data.reshape(30,-1)
#     reshaped_middle_layer_test_data.shape


# In[152]:

9200 / 20


# In[179]:

from sklearn.decomposition import PCA
from sklearn.lda import LDA


# In[185]:

clf = LDA()
lda_res = clf.fit_transform(middle_layer_test_data, test_target_gcd)
print lda_res.shape
plt.hist(lda_res[test_target_gcd == 1])
plt.show()
plt.hist(lda_res[test_target_gcd == 0])
plt.show()





