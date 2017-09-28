from importlib import reload

import numpy as np
import scipy.io as sio

def extractSpellingSequenceData(res):
    train_trial = res['mrk']['trial_idx'][0][0][0]
    train_block = res['mrk']['block_idx'][0][0][0]
    stimulus = res['mrk']['stimulus'][0][0][0]

    train_mode = np.zeros(res['mrk']['mode'][0][0].shape[1]).astype(np.int8)

    train_mode[np.where(res['mrk']['mode'][0][0][0] == 1)[0]] = 1
    train_mode[np.where(res['mrk']['mode'][0][0][1] == 1)[0]] = 2
    train_mode[np.where(res['mrk']['mode'][0][0][2] == 1)[0]] = 3

    return stimulus, train_trial, train_block, train_mode


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


def extractBasicDataTuBerlin(res):
    part_index = 1
    set_2 = [x for x in res['mnt']['clab']]

    set_1= [x for x in res['data'][part_index].channels]

    # electrode used in the experiment
    # set_2 = [x[0] for x in res['bbci']['bbci'][0][0]['analyze'][0][0]['features'][0][0]['clab'][0][0][0]]

    all_relevant_channels = []
    channels_names = []
    for i in range(len(set_1)):
        if (set_1[i] in set_2):

            all_relevant_channels.append(res['data'][part_index].X[:,i])
            channels_names.append(set_1[i])
    marker_positions = res['bbci_mrk'][part_index].event.trial_idx
    target = 2- res['data'][part_index].y


    train_trial = res['bbci_mrk'][part_index].event.trial_idx
    train_block = res['bbci_mrk'][part_index].event.block_idx
    stimulus = res['bbci_mrk'][part_index].event.stimulus

    train_mode = np.argmax(res['bbci_mrk'][part_index].event.mode,axis=1) + 1
    train_mode = train_mode.astype(np.int8)


    # return stimulus, train_trial, train_block, train_mode


    return all_relevant_channels, channels_names, marker_positions, target, stimulus, train_trial, train_block, train_mode


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

def extract_time_windows(multi_dim_time_series, sample_start_idx, sample_before, sample_after):
    number_of_samples = len(sample_start_idx)
    number_of_timestamp = sample_after - sample_before

    # by convention of the theano's scan, the channel is the last dimension:
    number_of_channels = multi_dim_time_series.shape[-1]
    return_value = np.zeros((number_of_samples, number_of_timestamp, number_of_channels), dtype='int16')

    timestamp_counter = 0
    for i in range(sample_before, sample_after):
        return_value[:, timestamp_counter, :] = multi_dim_time_series[sample_start_idx + i, :]
        timestamp_counter += 1

    return return_value


def extract_data_by_triggers(all_relevant_channels, marker_positions, target, ms_before, ms_after):
    all_target_transpose = np.asarray(all_relevant_channels).T
    before_trigger = int((ms_before * 1.0) / 5)
    after_trigger = int((ms_after * 1.0) / 5)

    all_data = extract_time_windows(all_target_transpose, marker_positions - 1, before_trigger, after_trigger)

    return all_data.astype('float32'), target


def create_data_rep_training_public(file_name, fist_time_stamp, last_time_stamp, downsample_params=1):
    """
    The function divide the data into epochs and shaping it such that it is
    easy to do operation per stimuli category on it.
    """
    data_from_file = readCompleteMatFile(file_name)
    data_for_eval = extract_data_by_triggers(data_from_file['all_relevant_channels'], data_from_file['marker_positions'],
                                             data_from_file['target'], fist_time_stamp, last_time_stamp)

    data_for_eval = (downsample_data_pub(data_for_eval[0], data_for_eval[0].shape[1], downsample_params), data_for_eval[1])

    train_mode_per_block = data_from_file['train_mode'].reshape(-1, 30)[:, 0]
    all_data_per_char_as_matrix = np.zeros(
        (train_mode_per_block.shape[0], 30, data_for_eval[0].shape[1], data_for_eval[0].shape[2]))
    all_data_per_char = dict()
    target_per_char_as_matrix = np.zeros((train_mode_per_block.shape[0], 30), dtype=np.int)
    for i, stimuli_i in enumerate(range(1, 31)):
        all_data_per_char[i] = data_for_eval[0][data_from_file['stimulus'] == stimuli_i]
        all_data_per_char_as_matrix[:, i, :, :] = data_for_eval[0][data_from_file['stimulus'] == stimuli_i]

    target_per_char = dict()
    for i, stimuli_i in enumerate(range(1, 31)):
        target_per_char[i] = data_for_eval[1][data_from_file['stimulus'] == stimuli_i]
        target_per_char_as_matrix[:, i] = data_for_eval[1][data_from_file['stimulus'] == stimuli_i]

    return all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix


def downsample_data_pub(data, number_of_original_samples, down_samples_param):
    new_number_of_time_stamps = number_of_original_samples / down_samples_param

    # print  data_for_eval
    temp_data_for_eval = np.zeros((data.shape[0], int(new_number_of_time_stamps), data.shape[2]))

    for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
        temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
    return temp_data_for_eval




def set_keras_backend(backend):
    import keras.backend as K
    import os

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

