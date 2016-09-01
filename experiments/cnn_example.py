from OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
import numpy as np
from keras.utils.np_utils import to_categorical
from scipy import stats

"""
In general, this code implement a Convolutional Neural network for classifying P300.
Please note that unlike image CNN this architecture first process the spatial domain and
only later the temporal domain.

Note also that in this example the ratio between target and non-target is 1:30,
this means that the accuracy will be ~97% if the model will always return non-target.
The current code only train for 1 epoch and receive a 97% accuracy (i.e. it is useless)


You are strongly encourage to read the article:

for reference, see:
http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=5492691&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D5492691



"""
def downsample_data(data, number_of_original_samples, down_samples_param):
    new_number_of_time_stamps = number_of_original_samples / down_samples_param

    temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))

    for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
        temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
    return temp_data_for_eval


def create_train_data(data_from_mat, down_samples_param, indexes):
    all_positive_train = []
    all_negative_train = []

    last_time_stamp = 800
    fist_time_stamp = -200

    data_for_eval = ExtractDataVer4(data_from_mat['all_relevant_channels'], data_from_mat['marker_positions'],
                                    data_from_mat['target'], fist_time_stamp, last_time_stamp)

    temp_data_for_eval = downsample_data(data_for_eval[0],data_for_eval[0].shape[1], down_samples_param)

    positive_train_data_gcd = temp_data_for_eval[
        np.all([indexes, data_from_mat['target'] == 1], axis=0)]
    negative_train_data_gcd = temp_data_for_eval[
        np.all([indexes, data_from_mat['target'] == 0], axis=0)]
    all_positive_train.append(positive_train_data_gcd)
    all_negative_train.append(negative_train_data_gcd)

    positive_train_data_gcd = np.vstack(all_positive_train)
    negative_train_data_gcd = np.vstack(all_negative_train)

    all_data = np.vstack([positive_train_data_gcd, negative_train_data_gcd])

    all_tags = np.vstack(
        [np.ones((positive_train_data_gcd.shape[0], 1)), np.zeros((negative_train_data_gcd.shape[0], 1))])
    categorical_tags = to_categorical(all_tags)

    shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags

if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D

    '''
    define the neural network model:
    '''
    model = Sequential()

    number_of_time_stamps =25
    number_of_out_channels = 10
    number_of_in_channels = 55
    length_of_time_axe_mask = 10



    model.add(Convolution2D(nb_filter=10,
                            nb_col=number_of_out_channels,
                            nb_row=1,
                            input_shape=(1, number_of_time_stamps, number_of_in_channels),
                            border_mode='same',
                            init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, number_of_in_channels)))
    model.add(Convolution2D(nb_filter=number_of_out_channels, nb_row=6, nb_col=1, border_mode='same',init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(20, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    # model.add(LSTM(input_dim=55, output_dim=20,return_sequences=False))
    # # model.add(Dense(275))
    # # model.add(Activation('tanh'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\RSVP_Color116msVPfat.mat'



    raw_data = readCompleteMatFile(file_name)
    train_data, train_tag = create_train_data(raw_data, 8, raw_data['train_mode'] == 1)

    train_data_single_color_channel = np.expand_dims(stats.zscore(train_data, axis = 1), axis = 1)

    test_data, test_tag = create_train_data(raw_data, 8, raw_data['train_mode'] != 1)
    model.fit(train_data_single_color_channel, train_tag, nb_epoch=1, show_accuracy=True)

    test_data_single_color_channel = np.expand_dims(stats.zscore(test_data, axis=1), axis=1)
    print model.evaluate(test_data_single_color_channel, test_tag, show_accuracy=True)
    print train_data.shape

