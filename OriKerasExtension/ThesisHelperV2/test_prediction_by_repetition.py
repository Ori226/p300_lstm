__author__ = 'ori22_000'

import unittest
from CompareToBlankerts import predict_by_repetitions4, PredictByRepetitions3
from OriKerasExtension.ThesisHelper import readCompleteMatFile


class MyTestCase(unittest.TestCase):
    def test_something(self):
        pass
        # res = readCompleteMatFile(r'C:\Users\ori22_000\Documents\Thesis\dataset_all\RSVP_Color116msVPgcd.mat');
        # print res.keys()
        # stimulue_vector = res['stimulus']
        #should return dictionary trial-> blocl-> stimuli-> index
        # predict_by_repetitions4()

        self.assertEqual(True, False)


def get_indexes_by_stimuli(stimulus_range, stimulus, train_trial, train_block):

    stimuli_histogram = {}
    for i in stimulus_range:
        stimuli_histogram[i] = dict(stimuli=i,
                                    idx=np.where(stimulus == i)[0],
                                    train_trial=train_trial[np.where(stimulus == i)[0]],
                                    train_block=train_block[np.where(stimulus == i)[0]])
    return stimuli_histogram


import numpy as np
if __name__ == '__main__':
    res = readCompleteMatFile(r'C:\Users\ori22_000\Documents\Thesis\dataset_all\RSVP_Color116msVPgcd.mat');

    print get_indexes_by_stimuli(range(1,3), res['stimulus'], res['train_trial'], res['train_block'])
    # # stimuli_histogram = {}
    # for i in range(1,31):
    #     stimuli_histogram[i] = dict(stimuli=i,
    #                                 idx=np.where(res['stimulus'] == i)[0],
    #                                 train_trial=res['train_trial'][np.where(res['stimulus'] == i)[0]],
    #                                 train_block=res['train_block'][np.where(res['stimulus'] == i)[0]])
        # np.where(res['stimulus'] == i)[0]
        # print np.where(res['stimulus'] == i)[0]
        # print res['train_trial'][np.where(res['stimulus'] == i)[0]]
        # print res['train_block'][np.where(res['stimulus'] == i)[0]]
        # print np.where(res['stimulus'] == i)[0]

    # print stimuli_histogram[2]['idx'][np.where(stimuli_histogram[2]['train_block'] <= 3)[0]]



    # stimulue_vector = res['stimulus']
    # PredictByRepetitions3(res,1,None)
    #should return dictionary trial-> blocl-> stimuli-> index
    # predict_by_repetitions4()


    # unittest.main()
