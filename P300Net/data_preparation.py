import numpy as np
import random
from OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
from experiments.P300_RSVP.common import downsample_data
import matplotlib.pyplot as plt;

def create_data_rep_training(file_name, fist_time_stamp, last_time_stamp, downsampe_params=1):
    """
    The function divide the data into epochs and shaping it such that it is
    easy to do operation per stimuli category on it.
    :param file_name:
    :param fist_time_stamp:
    :param last_time_stamp:
    :return:
    """
    gcd_res = readCompleteMatFile(file_name)
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    data_for_eval = (downsample_data(data_for_eval[0], data_for_eval[0].shape[1], downsampe_params), data_for_eval[1])

    train_mode_per_block = gcd_res['train_mode'].reshape(-1, 30)[:, 0]
    all_data_per_char_as_matrix = np.zeros(
        (train_mode_per_block.shape[0], 30, data_for_eval[0].shape[1], data_for_eval[0].shape[2]))
    all_data_per_char = dict()
    target_per_char_as_matrix = np.zeros((train_mode_per_block.shape[0], 30), dtype=np.int)
    for i, stimuli_i in enumerate(range(1, 31)):
        all_data_per_char[i] = data_for_eval[0][gcd_res['stimulus'] == stimuli_i]
        all_data_per_char_as_matrix[:, i, :, :] = data_for_eval[0][gcd_res['stimulus'] == stimuli_i]

    target_per_char = dict()
    for i, stimuli_i in enumerate(range(1, 31)):
        target_per_char[i] = data_for_eval[1][gcd_res['stimulus'] == stimuli_i]
        target_per_char_as_matrix[:, i] = data_for_eval[1][gcd_res['stimulus'] == stimuli_i]

    return all_data_per_char, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix



def _get_all_possible_combination(samples_indexes, max_number_of_repetitionm, repetition_in_model):
    from itertools import combinations
    all_combination = []
    for block_of_repetition_index in samples_indexes.reshape(-1, max_number_of_repetitionm):
        indexes = list(combinations(block_of_repetition_index, repetition_in_model))
        # now select 3 randomaly
        all_combination.extend(indexes)

    return all_combination


def triplet_data_generator(data, tags, batch_size, select=3, outof=10):
    from scipy import stats
    stimuli_category_size = 30
    number_of_repetition = select
    magic_number = number_of_repetition * stimuli_category_size
    number_of_samples = data.shape[0]
    time_samples_dim_size = data.shape[2]
    channel_dim_size = data.shape[3]

    all_combination = _get_all_possible_combination(np.arange(number_of_samples), outof, select)


    while True:
        shuffled_combination = np.random.permutation(all_combination)
        for counter_i ,i in enumerate(range(0,len(shuffled_combination), batch_size)):
            print "{}:{}".format(i, min(i +batch_size,len(shuffled_combination) ))
            # if counter_i == 3:
            #     break

            batch_data = np.zeros((magic_number, batch_size, time_samples_dim_size, channel_dim_size), dtype=np.float32)
            batch_tags = np.zeros((batch_size, stimuli_category_size), dtype=np.int8)
            counter = 0
            for single_combination in shuffled_combination[i:min(i +batch_size,len(shuffled_combination) )]:

                batch_data[:, counter, :, :] = np.vstack([data[item] for item in single_combination])
                batch_tags[counter] = np.mean(np.vstack([tags[item] for item in single_combination]), axis=0)
                counter += 1

            input_dict = dict(
                [["positive_item_input_{}".format(i), stats.zscore(batch_data[i], axis=1)] for i in
                 range(90)])

            input_dict['triplet_loss'] = batch_tags

            yield input_dict


def get_number_of_samples_per_epoch(number_of_samples,select=3, outof=10):
    dictionary_size = 30
    return len( _get_all_possible_combination(np.arange(number_of_samples), outof, select)) * dictionary_size

def triplet_data_generator_no_dict(data, tags, batch_size, select=3, outof=10):
    from scipy import stats
    stimuli_category_size = 30
    number_of_repetition = select
    magic_number = number_of_repetition * stimuli_category_size
    number_of_samples = data.shape[0]
    time_samples_dim_size = data.shape[2]
    channel_dim_size = data.shape[3]

    all_combination = _get_all_possible_combination(np.arange(number_of_samples), outof, select)


    while True:
        shuffled_combination = np.random.permutation(all_combination)
        for counter_i ,i in enumerate(range(0,len(shuffled_combination), batch_size)):
            # print "{}:{}".format(i, min(i +batch_size,len(shuffled_combination) ))
            if counter_i  == 5:
                break

            batch_data = np.zeros((batch_size, magic_number, time_samples_dim_size, channel_dim_size), dtype=np.float32)
            batch_tags = np.zeros((batch_size, magic_number ), dtype=np.int8)
            counter = 0
            for single_combination in shuffled_combination[i:min(i +batch_size,len(shuffled_combination) )]:

                batch_data[counter, :, :, :] = np.vstack([stats.zscore(data[item],axis=1) for item in single_combination])
                batch_tags[counter, :] = np.vstack([tags[item] for item in single_combination]).flatten()
                counter += 1

            # input_dict = dict(
            #     [["positive_item_input_{}".format(i), stats.zscore(batch_data[i], axis=1)] for i in
            #      range(90)])

            # input_dict['triplet_loss'] = batch_tags
            # noramlized_batch_data = stats.zscore(batch_data, axis=2)
            noramlized_batch_data = batch_data

            # yield noramlized_batch_data.reshape(-1, noramlized_batch_data.shape[2], noramlized_batch_data.shape[3]), batch_tags.T.flatten()
            yield noramlized_batch_data.reshape(noramlized_batch_data.shape[0]*noramlized_batch_data.shape[1],
                                                noramlized_batch_data.shape[2],noramlized_batch_data.shape[3]), batch_tags.flatten()
            # yield batch_tags.T.reshape(-1,1), batch_tags.T.reshape(-1,1)

def triplet_data_generator_no_dict_no_label(data, tags, batch_size, select=3, outof=10):
    from scipy import stats
    stimuli_category_size = 30
    number_of_repetition = select
    magic_number = number_of_repetition * stimuli_category_size
    number_of_samples = data.shape[0]
    time_samples_dim_size = data.shape[2]
    channel_dim_size = data.shape[3]

    all_combination = _get_all_possible_combination(np.arange(number_of_samples), outof, select)


    while True:
        shuffled_combination = np.random.permutation(all_combination)
        for counter_i ,i in enumerate(range(0,len(shuffled_combination), batch_size)):
            print "{}:{}".format(i, min(i +batch_size,len(shuffled_combination) ))
            if counter_i  == 5:
                break

            batch_data = np.zeros((magic_number, batch_size, time_samples_dim_size, channel_dim_size), dtype=np.float32)
            batch_tags = np.zeros((magic_number , batch_size), dtype=np.int8)
            counter = 0
            for single_combination in shuffled_combination[i:min(i +batch_size,len(shuffled_combination) )]:

                batch_data[:, counter, :, :] = np.vstack([data[item] for item in single_combination])
                batch_tags[:, counter] = np.vstack([tags[item] for item in single_combination]).flatten()
                counter += 1

            input_dict = dict(
                [["positive_item_input_{}".format(i), stats.zscore(batch_data[i], axis=1)] for i in
                 range(90)])

            input_dict['triplet_loss'] = batch_tags
            noramlized_batch_data = stats.zscore(batch_data, axis=2)

            # yield noramlized_batch_data.reshape(-1, noramlized_batch_data.shape[2], noramlized_batch_data.shape[3]), batch_tags.T.flatten()
            yield noramlized_batch_data.reshape(noramlized_batch_data.shape[0]*noramlized_batch_data.shape[1],
                                                noramlized_batch_data.shape[2]*noramlized_batch_data.shape[3])
            # yield batch_tags.T.reshape(-1,1), batch_tags.T.flatten()


def triplet_data_collection(data, tags, batch_size, select=3, outof=10):
    from scipy import stats
    stimuli_category_size = 30
    number_of_repetition = select
    magic_number = number_of_repetition * stimuli_category_size
    number_of_samples = data.shape[0]
    time_samples_dim_size = data.shape[2]
    channel_dim_size = data.shape[3]

    all_combination = _get_all_possible_combination(np.arange(number_of_samples), outof, select)
    shuffled_combination = np.random.permutation(all_combination)

    batch_data = np.zeros((magic_number, batch_size, time_samples_dim_size, channel_dim_size), dtype=np.float32)
    counter = 0
    for i in range(0,len(shuffled_combination), batch_size):



        batch_tags = np.zeros((batch_size, stimuli_category_size), dtype=np.int8)

        for single_combination in shuffled_combination[i:min(i +batch_size,len(shuffled_combination) )]:

            batch_data[:, counter, :, :] = np.vstack([data[item] for item in single_combination])
            batch_tags[counter] = np.mean(np.vstack([tags[item] for item in single_combination]), axis=0)
            counter += 1
            if counter == batch_size:
                input_dict = dict(
                    [["positive_item_input_{}".format(i), stats.zscore(batch_data[i], axis=1)] for i in
                     range(90)])

                input_dict['triplet_loss'] = batch_tags

                return input_dict

def triplet_data_collection_no_dict(data, tags, batch_size, select=3, outof=10):
    from scipy import stats
    stimuli_category_size = 30
    number_of_repetition = select
    magic_number = number_of_repetition * stimuli_category_size
    number_of_samples = data.shape[0]
    time_samples_dim_size = data.shape[2]
    channel_dim_size = data.shape[3]

    all_combination = _get_all_possible_combination(np.arange(number_of_samples), outof, select)
    shuffled_combination = np.random.permutation(all_combination)

    batch_data = np.zeros((magic_number, batch_size, time_samples_dim_size, channel_dim_size), dtype=np.float32)
    counter = 0
    for i in range(0,len(shuffled_combination), batch_size):



        batch_tags = np.zeros((batch_size, stimuli_category_size), dtype=np.int8)

        for single_combination in shuffled_combination[i:min(i +batch_size,len(shuffled_combination) )]:

            batch_data[:, counter, :, :] = np.vstack([data[item] for item in single_combination])
            batch_tags[counter] = np.mean(np.vstack([tags[item] for item in single_combination]), axis=0)
            counter += 1
            if counter == batch_size:
                input_dict = dict(
                    [["positive_item_input_{}".format(i), stats.zscore(batch_data[i], axis=1)] for i in
                     range(90)])

                input_dict['triplet_loss'] = batch_tags

                return batch_data.reshape(-1,25,55), batch_tags.reshape(-1,1)


if __name__ == "__main__":
    shuffled_combination = _get_all_possible_combination(np.arange(240,690), 10, 3)
    batch_size =4000
    for i in range(0, len(shuffled_combination), batch_size):
        print "{}:{}".format(i, min(i + batch_size, len(shuffled_combination)))

        for single_combination in shuffled_combination[i:min(i + batch_size, len(shuffled_combination))]:
            print single_combination

        # for block_of_repetition_index in np.random.permutation(np.where(train_mode_per_block)[0].reshape(-1, 10))[
        #                                  0:10]:




