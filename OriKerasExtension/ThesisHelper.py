import numpy as np
import scipy.io as sio
from scipy import stats


def readBlankertMatFile(file_path):
    res = sio.loadmat(file_path)
    # all electrode
    return extractBasicData(res)


def extractBasicData(res):
    set_1 = [x[0] for x in res['nfo']['clab'][0][0][0]]

    # electrode used in the experiment
    set_2 = [x[0] for x in res['bbci']['bbci'][0][0]['analyze'][0][0]['features'][0][0]['clab'][0][0][0]]

    all_relevant_channels = []
    channels_names = []
    for i in range(len(set_1)):
        if (set_1[i] in set_2):
            all_relevant_channels.append(res["ch%d" % (i + 1)].flatten())
            channels_names.append(set_1[i])
    marker_positions = res['mrk']['pos'][0][0][0]
    target = res['mrk']['y'][0][0][0]

    return all_relevant_channels, channels_names, marker_positions, target


def extractSpellingSequenceData(res):
    train_trial = res['mrk']['trial_idx'][0][0][0]
    train_block = res['mrk']['block_idx'][0][0][0]
    stimulus = res['mrk']['stimulus'][0][0][0]

    train_mode = np.zeros(res['mrk']['mode'][0][0].shape[1]).astype(np.int8)

    train_mode[np.where(res['mrk']['mode'][0][0][0] == 1)[0]] = 1
    train_mode[np.where(res['mrk']['mode'][0][0][1] == 1)[0]] = 2
    train_mode[np.where(res['mrk']['mode'][0][0][2] == 1)[0]] = 3

    return stimulus, train_trial, train_block, train_mode

def extract_2D_channel_location(file_path):
    res = sio.loadmat(file_path)
    set_1 = [x[0] for x in res['nfo']['clab'][0][0][0]]

    # electrode used in the experiment
    set_2 = [x[0] for x in res['bbci']['bbci'][0][0]['analyze'][0][0]['features'][0][0]['clab'][0][0][0]]

    valid_electrode_logical = np.in1d(np.array([x[0] for x in res['mnt']['clab'][0][0][0]]), np.array(set_2))
    valid_electrode_location_x_y = np.hstack([res['mnt']['x'][0][0], res['mnt']['y'][0][0]])[valid_electrode_logical, :]
    return valid_electrode_location_x_y


def readCompleteMatFile(file_path):
    """
    see 'Gaze-independent BCI-spelling using rapid serial visual presentation (RSVP)' to understand the experiment better
    :param file_path:
    :return: dictionary containing the following fields:
    - all_relevant_channels - the EEG data itself. A list of numpy array vector, each in the size of the complete
     recording
    - channels_names -  the EEG sensor location  name
    - marker_positions - the index of the time stamp in which the stimuli appear to the subject
    - target - a binary vector in the length of 'marker_positions' position field. 1 represent target
    stimuli and 0 represent non-target stimuli
    - train_trial - the index of the letter that is spelled (i.e if the subject is request to spell 'abc'
    there will be 3 trials (which are duplicate due to the repetition)
    - train_block - the index of the repetition (i.e. in Blankertz RSVP - between 1 to 10)
    - train_mode - represent whether the recording is calibration, copy spelling or free spell (see the article)
    """
    res = sio.loadmat(file_path)

    all_relevant_channels, channels_names, marker_positions, target = extractBasicData(res)
    stimulus, train_trial, train_block, train_mode = extractSpellingSequenceData(res)

    return_value = {'all_relevant_channels': all_relevant_channels,
                    'channels_names': channels_names,
                    'marker_positions': marker_positions,
                    'target': target,
                    'stimulus': stimulus,
                    'train_trial': train_trial,
                    'train_block': train_block,
                    'train_mode': train_mode}

    return return_value


import matplotlib.pyplot as plt
import random


def LoadSingleSubjectPython(file_name):
    res = readCompleteMatFile(file_name);

    all_data, all_tags = ExtractDataVer2(res['all_relevant_channels'], res['marker_positions'], res['target'], 0, 400);

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

def LoadSingleSubjectPythonNoPermute(file_name):
    res = readCompleteMatFile(file_name);

    all_data, all_tags = ExtractDataVer3(res['all_relevant_channels'], res['marker_positions'], res['target'], 0, 400);

    trasposed_data = all_data.transpose(0, 2, 1)

    trasposed_data = trasposed_data.reshape(trasposed_data.shape[0], -1)

    all_target = trasposed_data[np.where(all_tags == 1)[0], :]
    all_non_target = trasposed_data[np.where(all_tags != 1)[0], :]

    subset_size = all_target.shape[0]

    all_target = all_target
    all_non_target = all_non_target
    return [all_target, all_non_target, res['marker_positions']]


def LoadSingleSubjectPythonByMode(file_name, mode_number):
    res = readCompleteMatFile(file_name);

    mode_idx = np.where(res['train_mode'] == mode_number)[0]

    all_data, all_tags = ExtractDataVer2(res['all_relevant_channels'], res['marker_positions'][mode_idx],
                                         res['target'][mode_idx], 0, 400);

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


def LoadSingleSubjectPythonByMode2(file_name, mode_number, non_target_factor=1):
    res = readCompleteMatFile(file_name);

    mode_idx = np.where(res['train_mode'] == mode_number)[0]

    all_data, all_tags = ExtractDataVer2(res['all_relevant_channels'], res['marker_positions'][mode_idx],
                                         res['target'][mode_idx], 0, 400);

    trasposed_data = all_data.transpose(0, 2, 1)

    trasposed_data = trasposed_data.reshape(trasposed_data.shape[0], -1)

    all_target = trasposed_data[np.where(all_tags == 1)[0], :]
    all_non_target = trasposed_data[np.where(all_tags != 1)[0], :]

    subset_size = all_target.shape[0]
    np.random.seed(0)

    all_target = np.random.permutation(all_target)[0:subset_size,
                 :]  # FromFileListToArray(target_files, 'all_target_flatten', 600)
    all_non_target = np.random.permutation(all_non_target)[0:subset_size * non_target_factor,
                     :]  # FromFileListToArray(non_target_files, 'all_non_target_flatten',600 )
    return [all_target, all_non_target]


def ExtractDataVer1(all_relevant_channels, marker_positions, target):
    target_idx = marker_positions[np.where(target == 1)[0]] - 1

    target_idx = np.random.permutation(target_idx)
    train_target_idx = target_idx[0:500];
    test_target_idx = target_idx[500:];

    all_target_transpose = np.asarray(all_relevant_channels).T

    # *(1000.0/res['nfo']['fs'][0][0][0][0])

    number_of_samples = len(train_target_idx)
    before_trigger = int(100.0 / 5)
    after_trigger = int(1200.0 / 5)
    number_of_timestamp = before_trigger + after_trigger
    number_of_channels = len(all_relevant_channels)
    all_target_data = np.zeros((number_of_samples * 9, number_of_timestamp, number_of_channels))

    samples_counter = 0
    for i in range(len(train_target_idx)):
        for aug_i in range(-4, 5):
            #         print (aug_i )
            all_target_data[samples_counter, :, :] = all_target_transpose[
                                                     range(train_target_idx[i] - before_trigger - aug_i,
                                                           (train_target_idx[i] + after_trigger - aug_i)), :]
            samples_counter = samples_counter + 1

    test_all_target_data = np.zeros((len(test_target_idx), number_of_timestamp, number_of_channels))

    samples_counter = 0
    for i in range(len(test_target_idx)):
        test_all_target_data[samples_counter, :, :] = all_target_transpose[
                                                      range(test_target_idx[i] - before_trigger - aug_i,
                                                            (test_target_idx[i] + after_trigger - aug_i)), :]
        samples_counter = samples_counter + 1

    non_target_idx = marker_positions[np.where(target == 0)[0]] - 1
    number_of_samples = len(non_target_idx)

    non_target_idx = np.random.permutation(non_target_idx)

    train_sub_non_target = non_target_idx[0:all_target_data.shape[0]];
    test_sub_non_target = non_target_idx[
                          all_target_data.shape[0]:(all_target_data.shape[0] + test_target_idx.shape[0])];
    all_non_target_data = np.zeros((len(train_sub_non_target), number_of_timestamp, number_of_channels))

    for i in range(len(train_sub_non_target)):
        all_non_target_data[i, :, :] = all_target_transpose[range(train_sub_non_target[i] - before_trigger,
                                                                  (train_sub_non_target[i] + after_trigger)), :]

    test_all_non_target_data = np.zeros((len(test_sub_non_target), number_of_timestamp, number_of_channels))
    for i in range(len(test_sub_non_target)):
        test_all_non_target_data[i, :, :] = all_target_transpose[range(test_sub_non_target[i] - before_trigger,
                                                                       (test_sub_non_target[i] + after_trigger)), :]

    all_data = np.vstack((stats.zscore(all_target_data, axis=1), stats.zscore(all_non_target_data, axis=1)))
    all_tags = np.vstack((np.ones((all_target_data.shape[0], 1)), np.zeros((all_non_target_data.shape[0], 1))))

    all_test_data = np.vstack(
        (stats.zscore(test_all_target_data, axis=1), stats.zscore(test_all_non_target_data, axis=1)))
    all_test_tags = np.vstack(
        (np.ones((test_all_target_data.shape[0], 1)), np.zeros((test_all_non_target_data.shape[0], 1))))

    return all_data, all_tags, all_test_data, all_test_tags


def extractDataByFromTimeSeries(multi_dim_time_series, sample_start_idx, sample_before, sample_after):
    return multi_dim_time_series[sample_start_idx - sample_before: sample_start_idx + sample_after, :];


def extractTimeWindowFast(multi_dim_time_series, sample_start_idx, sample_before, sample_after):
    number_of_samples = len(sample_start_idx)
    number_of_timestamp = sample_after - sample_before

    # by convention of the theano's scan, the channel is the last dimension:
    number_of_channels = multi_dim_time_series.shape[-1]
    return_value = np.zeros((number_of_samples, number_of_timestamp, number_of_channels), dtype='int16')

    timestamp_counter = 0
    for i in range(sample_before, sample_after):
        return_value[:, timestamp_counter, :] = multi_dim_time_series[sample_start_idx + i, :]
        timestamp_counter = timestamp_counter + 1

    return return_value


def ExtractDataVer2(all_relevant_channels, marker_positions, target, ms_before, ms_after):
    target_idx = marker_positions[np.where(target == 1)[0]] - 1

    all_target_transpose = np.asarray(all_relevant_channels).T

    number_positive_of_samples = len(target_idx)
    before_trigger = int((ms_before * 1.0) / 5)
    after_trigger = int((ms_after * 1.0) / 5)

    all_target_data = extractTimeWindowFast(all_target_transpose, target_idx, before_trigger, after_trigger)

    non_target_idx = marker_positions[np.where(target == 0)[0]] - 1
    number_positive_of_samples = len(non_target_idx)

    all_non_target_data = extractTimeWindowFast(all_target_transpose, non_target_idx, before_trigger, after_trigger)




    # normalize the data over the time axe
    all_data = np.vstack((stats.zscore(all_target_data, axis=1).astype('float32'),
                          stats.zscore(all_non_target_data, axis=1).astype('float32')))
    all_tags = np.vstack((np.ones((all_target_data.shape[0], 1), dtype='int8'),
                          np.zeros((all_non_target_data.shape[0], 1), dtype='int8')))

    return all_data, all_tags


def ExtractDataVer3(all_relevant_channels, marker_positions, target, ms_before, ms_after):
    all_target_transpose = np.asarray(all_relevant_channels).T
    before_trigger = int((ms_before * 1.0) / 5)
    after_trigger = int((ms_after * 1.0) / 5)

    all_data = extractTimeWindowFast(all_target_transpose, marker_positions - 1, before_trigger, after_trigger)

    return stats.zscore(all_data, axis=1).astype('float32'), target


def ExtractDataVer4(all_relevant_channels, marker_positions, target, ms_before, ms_after):
    all_target_transpose = np.asarray(all_relevant_channels).T
    before_trigger = int((ms_before * 1.0) / 5)
    after_trigger = int((ms_after * 1.0) / 5)

    all_data = extractTimeWindowFast(all_target_transpose, marker_positions - 1, before_trigger, after_trigger)

    return all_data.astype('float32'), target



