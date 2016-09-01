import numpy as np

__author__ = 'ORI'



def create_target_table(result_dictionary, data_to_sort):
    all_blocks_and_trials = np.zeros(
        (len(np.unique(result_dictionary['train_trial'])) * len(np.unique(result_dictionary['train_block'])), 2))
    couter = 0
    number_of_trials = len(np.unique(result_dictionary['train_trial']))
    number_of_blocks = len(np.unique(result_dictionary['train_block']))
    number_of_stimulies = len(np.unique(result_dictionary['stimulus']))
    for uniqu_trial in np.unique(result_dictionary['train_trial']):
        for uniqu_block in np.unique(result_dictionary['train_block']):
            all_blocks_and_trials[couter, 0] = uniqu_trial
            all_blocks_and_trials[couter, 1] = uniqu_block
            couter += 1

    # should be 620x30
    sorted_indexes = np.sort(np.argsort(result_dictionary['stimulus']).reshape(30, -1), axis=1).flatten()
    all_table_targets = data_to_sort[sorted_indexes].reshape(30, -1).T
    all_data_for_sum = np.zeros((number_of_trials, number_of_blocks, number_of_stimulies))
    for i in range(number_of_trials):
        all_data_for_sum[i] = all_table_targets[i * 10:(i + 1) * 10]

    all_data = np.hstack([np.asarray(all_blocks_and_trials), all_table_targets])



    columns = {0: 'train_trial', 1: 'train_block'}
    for i, stimuli in enumerate(np.unique(result_dictionary['stimulus'])):
        columns[i + 2] = stimuli

    return all_data, columns, all_data_for_sum


def accuracy_by_repetition(target_vector, probability_vector, number_of_repetition=10):
    gt = np.argmax(np.sum(target_vector[:, :number_of_repetition, :], axis=1),axis=1)
    actual = np.argmax(np.sum(probability_vector[:, :number_of_repetition, :], axis=1), axis=1)
    accuracy = 1.0 * np.sum(gt == actual) / gt.shape[0] * 1.0
    return accuracy