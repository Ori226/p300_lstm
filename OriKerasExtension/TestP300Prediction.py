import sys

from P300Prediction import create_target_table, accuracy_by_repetition


sys.path.append(r'C:\Users\ORI\Documents\IDC-non-sync\Thesis\PythonApplication1\OriKerasExtension')

import pandas as pd
import unittest

from ThesisHelper import readCompleteMatFile
import numpy as np


class TestTestP300Prediction(unittest.TestCase):

    def test_a(self):
        gcd_res = readCompleteMatFile(r'C:\Users\ORI\Documents\Thesis\dataset_all\RSVP_Color116msVPicr.mat');
        all_data, columns, all_data_for_sum = create_target_table(gcd_res, gcd_res['target'])
        results_table = pd.DataFrame( all_data)
        results_table.rename(columns=columns, inplace=True)
        # print results_table.columns
        results_table.to_csv('res.csv')
        # print results_table

        #now get a vector of random probabilities:


        temp = np.random.rand(len(gcd_res['target']),1)

        all_data2, columns2, all_data_for_sum2 = create_target_table(gcd_res, temp)



        # sum over the trials axe compare
        gt = np.argmax(np.sum(all_data_for_sum, axis=1),axis=1)
        actual = np.argmax(np.sum(all_data_for_sum2, axis=1),axis=1)
        print accuracy_by_repetition(all_data_for_sum, all_data_for_sum2)



        results_table2 = pd.DataFrame( all_data2)
        results_table2.rename(columns=columns2, inplace=True)
        # print results_table.columns
        results_table2.to_csv('res2.csv')



        temp2 = results_table2[all_data2[:, 1]<3, :]
        pass





        # self.fail("Not implemented")


if __name__ == '__main__':
    unittest.main()
