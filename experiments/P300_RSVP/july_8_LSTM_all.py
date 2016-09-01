from experiments.P300_RSVP.common import *

__author__ = 'ORI'

import numpy as np
import os
import cPickle as pickle

from   OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
import OriKerasExtension.P300Prediction

reload(OriKerasExtension.P300Prediction)
from OriKerasExtension.P300Prediction import accuracy_by_repetition, create_target_table

rng = np.random.RandomState(42)

#
# def create_evaluation_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1):
#     data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'], gcd_res['target'],
#                                     fist_time_stamp, last_time_stamp)
#
#     temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)
#
#     test_data_gcd, test_target_gcd = temp_data_for_eval[gcd_res['train_mode'] != 1], data_for_eval[1][
#         gcd_res['train_mode'] != 1]
#     return test_data_gcd, test_target_gcd
#
#
# def downsample_data(data, number_of_original_samples, down_samples_param):
#     new_number_of_time_stamps = number_of_original_samples / down_samples_param
#
#     # print  data_for_eval
#     temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))
#
#     for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
#         temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
#     return temp_data_for_eval
#
#
# def create_training_and_testing(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1,
#                                 take_same_number_positive_and_negative=False):
#     train_data, train_tags = create_train_data(gcd_res, fist_time_stamp, last_time_stamp, down_samples_param)
#     test_data, test_tags = create_evaluation_data(gcd_res=gcd_res, fist_time_stamp=fist_time_stamp,
#                                                   last_time_stamp=last_time_stamp,
#                                                   down_samples_param=down_samples_param)
#     func_args = dict(fist_time_stamp=fist_time_stamp, last_time_stamp=last_time_stamp,
#                      down_samples_param=down_samples_param,
#                      take_same_number_positive_and_negative=take_same_number_positive_and_negative)
#     return train_data, train_tags, test_data, test_tags, func_args
#
#
# def create_train_data(gcd_res, fist_time_stamp=0, last_time_stamp=400, down_samples_param=1):
#     data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
#                                     gcd_res['target'], fist_time_stamp, last_time_stamp)
#
#     temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)
#     all_tags = gcd_res['target'][gcd_res['train_mode'] == 1]
#     all_data = temp_data_for_eval[gcd_res['train_mode'] == 1]
#     return all_data, all_tags
#
#
# def create_data_for_compare_by_repetition(file_name):
#     gcd_res = readCompleteMatFile(file_name)
#     sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
#                        train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
#                        stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])
#     return sub_gcd_res
#
#
# class EvaluateByRepetition(object):
#     def __init__(self, subject_file):
#         super(EvaluateByRepetition, self).__init__()
#         self.sub_gcd_res = create_data_for_compare_by_repetition(file_name)
#
#     def foo(self, actual, prediction):
#         _, _, gt_data_for_sum = create_target_table(self.sub_gcd_res, actual)
#         _, _, actual_data_for_sum = create_target_table(self.sub_gcd_res, prediction[:, 1])
#
#         all_accuracies = dict([
#                                   [rep, accuracy_by_repetition(actual_data_for_sum, gt_data_for_sum,
#                                                                number_of_repetition=rep)]
#                                   for rep in range(1,11)])
#
#         print ", ".join([
#                             "acc {}:{}".format(k, v)
#                             for k, v in all_accuracies.iteritems()])
#         return all_accuracies


if __name__ == "__main__":
    model_20 = None
    model_100 = None

    all_subjects = ["RSVP_Color116msVPpia.mat",
                    "RSVP_Color116msVPicr.mat",
                    "RSVP_Color116msVPfat.mat",
                    "RSVP_Color116msVPgcb.mat",
                    "RSVP_Color116msVPgcc.mat",
                    "RSVP_Color116msVPgcd.mat",
                    "RSVP_Color116msVPgcf.mat",
                    "RSVP_Color116msVPgcg.mat",
                    "RSVP_Color116msVPgch.mat",
                    "RSVP_Color116msVPiay.mat",
                    "RSVP_Color116msVPicn.mat"];

    # all_subjects = ["RSVP_Color116msVPicr.mat"]

    data_base_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all'
    # model = LDA()

    all_models = [CNN_MLP(50)]
    for model_type in all_models:

        all_model_results = []

        for subject in all_subjects:
            file_name = os.path.join(data_base_dir, subject)
            gcd_res = readCompleteMatFile(file_name)
            repetition_eval = EvaluateByRepetition(file_name)

            for data_extraction_method in [create_training_and_testing(gcd_res, -3000, 800, 1, False)
                                           ]:
                # create_training_and_testing(gcd_res, 0, 400, 1, True)
                training_data, train_tags, testing_data, test_tags, func_args = data_extraction_method
                model = model_type  # type: GeneralModel
                print "starting {}:{}:{}".format(subject, model.get_name()[-7:-4],
                                                 ",".join([str(x) for x in func_args.values()]))

                # training_data = create_train_data(gcd_res, 0, 400, 1, True)
                # testing_data = create_evaluation_data(gcd_res, 0, 400, 1)

                # training_data, train_tags, testing_data, test_tags = create_training_and_testing(gcd_res, 0, 400, 1, True)

                # model = My_LDA()
                model.fit(training_data, train_tags)
                prediction_res = model.predict(testing_data)
                all_accuracies = repetition_eval.foo(test_tags, prediction_res)
                all_model_results.append(
                    dict(all_accuracies=all_accuracies, subject_name=subject, model=model.get_name(),
                         model_params=model.get_params(), func_args=func_args))
                model.reset()
                print "end {}:{}:{}".format(subject, model.get_name()[-7:-4],
                                            ",".join([str(x) for x in func_args.values()]))

        pickle.dump(all_model_results, file=open(model_type.get_name() + "_b.p", "wb"))

    pass
