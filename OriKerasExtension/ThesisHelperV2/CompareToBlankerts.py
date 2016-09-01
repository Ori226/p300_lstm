import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append(r"../")
from OriKerasExtension.ThesisHelper import readBlankertMatFile, readCompleteMatFile, LoadSingleSubjectPython, \
    ExtractDataVer2, LoadSingleSubjectPythonByMode

'''
general idea:
compare the success by averaging the differnet attempts

'''


def PredictByRepetitions3(loaded_data, num_of_rep, predictions):
    indexs = np.logical_and(loaded_data['train_mode'] != 1, loaded_data['train_block'] <= num_of_rep)
    tags = loaded_data['target']
    y_test = tags[np.where(indexs)[0]]

    ALPHABET_SIZE = 30

    print ('start aggregating')
    '''
    mean over same trial
    '''

    counter = 0
    true_counter = 0;
    false_counter = 0
    for trial_i in np.unique(loaded_data['train_trial']):
        print ("trial_i:", trial_i)
        relevant_idx = loaded_data['train_trial'] == trial_i
        if (len(np.where(relevant_idx)[0]) == 0):
            continue

        relevant_tags = np.logical_and(relevant_idx, tags == 1)
        tag_res = loaded_data['stimulus'][np.where(relevant_tags)[0]]
        # print ("tag_res",tag_res[0][0])
        prediction_histogram = np.zeros((num_of_rep, ALPHABET_SIZE))
        for block_i in range(num_of_rep):
            relevant_idx_per_block = np.logical_and(relevant_idx, loaded_data['train_block'] == (block_i + 1))
            print (np.where(relevant_idx_per_block))
            relevaant_stimulies = loaded_data['train_trial_stimulus'][np.where(relevant_idx_per_block), :][0]
            selected = relevaant_stimulies[np.argmax(predictions[np.where(relevant_idx_per_block), :][0].flatten())]
            print ("selected", selected)
            print("argmax", np.argmax(predictions[np.where(relevant_idx_per_block), :][0].flatten()))
            prediction_histogram[
                block_i, selected - 1] = 1  # predictions[np.where(relevant_idx_per_block),:][0].flatten()

        mean_prediction_histogram = np.mean(prediction_histogram, axis=0)

        predicted_value = np.argmax(mean_prediction_histogram) + 1
        gt_value = tag_res[0][
            0]  # np.argmax(tags[np.where(np.logical_and(relevant_idx, loaded_data['train_block'] == 1))[0],:])

        print("prediction:", predicted_value, " gt: ", gt_value)

        if predicted_value == gt_value:
            true_counter +=  1
        else:
            false_counter += 1

        if (False):
            plt.bar(range(30), mean_prediction_histogram)
            plt.show()
            plt.bar(range(30), tags[np.where(np.logical_and(relevant_idx, loaded_data['train_block'] == 1))[0], :])
            plt.show()
            print ("mean_prediction_histogram:", mean_prediction_histogram)

    print ("true_counter: ", true_counter, " false_counter", false_counter)
    return np.where(indexs)[0]


def get_indexes_by_stimuli(stimulus_range, stimulus, train_trial, train_block):

    stimuli_histogram = {}
    for i in stimulus_range:
        stimuli_histogram[i] = dict(stimuli=i,
                                    idx=np.where(stimulus == i)[0],
                                    train_trial=train_trial[np.where(stimulus == i)[0]],
                                    train_block=train_block[np.where(stimulus == i)[0]])
    return stimuli_histogram



def estimate_model(prediction, stimulus, train_trial, train_block):
    """

    :param prediction: prediction in the same order as the stimulus order
    :param stimulus:
    :param train_trial:
    :param train_block:
    :return:
    """


    stimuli_histogram = {}
    for i in stimulus_range:
        stimuli_histogram[i] = dict(stimuli=i,
                                    idx=np.where(stimulus == i)[0],
                                    train_trial=train_trial[np.where(stimulus == i)[0]],
                                    train_block=train_block[np.where(stimulus == i)[0]])
    return stimuli_histogram


def predict_by_repetitions4(trial_vector, stimuli_vector, repetition_vector , predictions):
    """

    :param trial_vector:
    :param repetition_vector:
    :param predictions:
    :return: for each trial the histogram of decision
    """

    distinct_trials = set(trial_vector)
    for trial_i in np.unique(trial_vector):
        pass
        # group indexes by their stimuli



    # indexs = np.logical_and(loaded_data['train_mode'] != 1, loaded_data['train_block'] <= num_of_rep)
    # tags = loaded_data['target']
    # y_test = tags[np.where(indexs)[0]]
    #
    # ALPHABET_SIZE = 30
    #
    # print ('start aggregating')
    # '''
    # mean over same trial
    # '''
    #
    # counter = 0
    # true_counter = 0;
    # false_counter = 0
    # for trial_i in np.unique(loaded_data['train_trial']):
    #     print ("trial_i:", trial_i)
    #     relevant_idx = loaded_data['train_trial'] == trial_i
    #     if (len(np.where(relevant_idx)[0]) == 0):
    #         continue
    #
    #     relevant_tags = np.logical_and(relevant_idx, tags == 1)
    #     tag_res = loaded_data['stimulus'][np.where(relevant_tags)[0]]
    #     # print ("tag_res",tag_res[0][0])
    #     prediction_histogram = np.zeros((num_of_rep, ALPHABET_SIZE))
    #     for block_i in range(num_of_rep):
    #         relevant_idx_per_block = np.logical_and(relevant_idx, loaded_data['train_block'] == (block_i + 1))
    #         print (np.where(relevant_idx_per_block))
    #         relevaant_stimulies = loaded_data['train_trial_stimulus'][np.where(relevant_idx_per_block), :][0]
    #         selected = relevaant_stimulies[np.argmax(predictions[np.where(relevant_idx_per_block), :][0].flatten())]
    #         print ("selected", selected)
    #         print("argmax", np.argmax(predictions[np.where(relevant_idx_per_block), :][0].flatten()))
    #         prediction_histogram[
    #             block_i, selected - 1] = 1  # predictions[np.where(relevant_idx_per_block),:][0].flatten()
    #
    #     mean_prediction_histogram = np.mean(prediction_histogram, axis=0)
    #
    #     predicted_value = np.argmax(mean_prediction_histogram) + 1
    #     gt_value = tag_res[0][
    #         0]  # np.argmax(tags[np.where(np.logical_and(relevant_idx, loaded_data['train_block'] == 1))[0],:])
    #
    #     print("prediction:", predicted_value, " gt: ", gt_value)
    #
    #     if predicted_value == gt_value:
    #         true_counter +=  1
    #     else:
    #         false_counter += 1
    #
    #     if (False):
    #         plt.bar(range(30), mean_prediction_histogram)
    #         plt.show()
    #         plt.bar(range(30), tags[np.where(np.logical_and(relevant_idx, loaded_data['train_block'] == 1))[0], :])
    #         plt.show()
    #         print ("mean_prediction_histogram:", mean_prediction_histogram)
    #
    # print ("true_counter: ", true_counter, " false_counter", false_counter)
    # return np.where(indexs)[0]



def foo():
    return 1, 2, 3


if __name__ == "__main__":



    # [all_target1, all_non_target1] = LoadSingleSubjectPythonByMode(r"C:\Users\ori22_000\Documents\Thesis\dataset\VPfat_11_01_24\RSVP_Color116msVPfat.mat" ,1)

    # [all_target2, all_non_target2] = LoadSingleSubjectPythonByMode(r"C:\Users\ori22_000\Documents\Thesis\dataset\VPfat_11_01_24\RSVP_Color116msVPfat.mat" ,2)

    # [all_target3, all_non_target3] = LoadSingleSubjectPythonByMode(r"C:\Users\ori22_000\Documents\Thesis\dataset\VPfat_11_01_24\RSVP_Color116msVPfat.mat" ,3)

    # LoadSingleSubjectPython(r"C:\Users\ori22_000\Documents\Thesis\dataset\VPfat_11_01_24\RSVP_Color116msVPfat.mat");

    temp = readCompleteMatFile(r"C:\Users\ori22_000\Documents\Thesis\dataset\VPfat_11_01_24\RSVP_Color116msVPfat.mat")
    PredictByRepetitions3(temp, 3, None);

    all_relevant_channels, channels_names, marker_positions, target = readCompleteMatFile(
        r"C:\Users\ori22_000\Documents\Thesis\dataset\VPfat_11_01_24\RSVP_Color116msVPfat.mat")
    # other option: take the model

    # load the trial data

    # load the prediction data - the prediction data is an index with tag\probablity


    a, b = foo()
    print ('hello');
