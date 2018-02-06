
from __future__ import print_function
import argparse

from scipy.stats import stats
from sklearn import cross_validation

from datasets.load_datasets import download_and_cache_file
from models import get_only_P300_model_LSTM_CNN, My_LDA_public, get_only_P300_model_CNN, get_only_P300_model_LSTM
from utils import create_data_rep_training_public
from sklearn.metrics import roc_auc_score
import numpy as np
import keras
__author__ = 'ORI'
np.random.seed(42)
rng = np.random.RandomState(42)
import os

def predict_using_model(model, data, tags):
    """
    The function calculate the performance of a model on a given data. Two metrics are calculated:
    1) spelling accuracy after 10 repetition for each letter. The letter with the highest
    probability is the predicted letter.
    2) Area under the curve(AUC). The roc-auc score is calculated on all the predictions, without aggregating prediction
    from the same trial.

    :param model: An instance of an object with 'predict' function. The predict return a value between 0 to 1 for each stimulu
    :param data: The data on which the model is doing the prediction
    :param tags: The ground truth label of each stimuli (i.e. target vs non-target
    :return accuracy, auc_score:
    """

    all_prediction_P300Net = model.predict(data)
    actual = np.argmax(np.mean(all_prediction_P300Net.reshape((-1, 10, 30)), axis=1), axis=1);
    gt = np.argmax(np.mean(tags.reshape((-1, 10, 30)), axis=1), axis=1)
    accuracy = np.sum(actual == gt) / float(len(gt))
    auc_score = roc_auc_score(tags.flatten(), all_prediction_P300Net)
    return accuracy, auc_score


def get_model(model_name, eeg_sample_shape):
    """

    :param model_name: name of the model
    :param eeg_sample_shape: the size of a single input entry
    :return: a an instance of a class that have a 'fit' function
    """
    if model_name == 'LDA':  # the only non-neural model
        model = My_LDA_public()
    else:
        if model_name == 'lstm_small':
            model = get_only_P300_model_LSTM(eeg_sample_shape, number_of_hidden=30)
        elif model_name == 'lstm_big':
            model = get_only_P300_model_LSTM(eeg_sample_shape, number_of_hidden=100)
        elif model_name == 'lstm_cnn_small':
            model = get_only_P300_model_LSTM_CNN(eeg_sample_shape, number_of_hidden=30)
        elif model_name == 'lstm_cnn_big':
            model = get_only_P300_model_LSTM_CNN(eeg_sample_shape, number_of_hidden=100)
        elif model_name == 'CNN':
            model = get_only_P300_model_CNN(eeg_sample_shape)


        from keras.optimizers import RMSprop
        model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'], )
    return model


def prepare_data_for_experiment(all_subjects, add_time_domain_noise, current_experiment_setting,
                                downsample_params,
                                number_of_k_fold, cross_validation_iter):
    """
    prepare the data for the experiment.
    """

    train_data_all_subject = []
    test_data_all_subject = []
    train_tags_all_subject = []
    test_tags_all_subject = []
    test_data_all_subject_with_noise = dict()

    for experiment_counter, subject in enumerate(all_subjects):
        print("start subject:{}".format(subject))
        file_name = download_and_cache_file(subject, experiment_suffix=current_experiment_setting)

        _, target_per_char, train_mode_per_block, all_data_per_char_as_matrix, target_per_char_as_matrix = \
            create_data_rep_training_public(file_name, -200, 800, downsample_params=downsample_params)
        print("done reading mat file")

        noise_data = dict()
        if add_time_domain_noise:
            noise_shifts = [-120, -80, -40, 0, 40, 80, 120]
        else:
            noise_shifts = [0]

        for time_shift_noise in noise_shifts:
            _, _, _, noise_data[time_shift_noise], _ = create_data_rep_training_public(
                file_name, (-200 + time_shift_noise), (800 + time_shift_noise), downsample_params=downsample_params)

        for rep_per_sub, cross_validation_indexes in enumerate(
                list(cross_validation.KFold(int(len(train_mode_per_block) / 10), n_folds=number_of_k_fold,
                                            random_state=42, shuffle=True))):
            if rep_per_sub < cross_validation_iter:
                continue

            def flatten_repetitions(data_to_flatten):
                return np.reshape(np.reshape(data_to_flatten.T * 10, (-1, 1)) + np.arange(10), (-1))

            train_indexes = flatten_repetitions(cross_validation_indexes[0])
            test_indexes = flatten_repetitions(cross_validation_indexes[1])

            train_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[train_indexes]).astype(np.float32))
            test_data_all_subject.append(np.asarray(all_data_per_char_as_matrix[test_indexes]).astype(np.float32))

            for time_shift_noise in noise_shifts:
                if time_shift_noise not in test_data_all_subject_with_noise:
                    test_data_all_subject_with_noise[time_shift_noise] = []
                test_data_all_subject_with_noise[time_shift_noise].append(
                    np.asarray(noise_data[time_shift_noise][test_indexes]).astype(np.float32))

            train_tags_all_subject.append(target_per_char_as_matrix[train_indexes])
            test_tags_all_subject.append(target_per_char_as_matrix[test_indexes])
            break

    # normalize the data
    train_data = stats.zscore(np.vstack(train_data_all_subject), axis=2)
    train_tags = np.vstack(train_tags_all_subject).flatten()
    test_data_with_noise = dict()
    for time_shift_noise in noise_shifts:
        test_data_with_noise[time_shift_noise] = stats.zscore(
            np.vstack(test_data_all_subject_with_noise[time_shift_noise]), axis=2)

    test_tags = np.vstack(test_tags_all_subject).flatten()
    return train_data, train_tags, test_data_with_noise, test_tags, noise_shifts


def train_and_evaluate(all_subjects, current_experiment_setting,
                       downsample_params,
                       add_time_domain_noise, cross_validation_iter,
                       number_of_k_fold,
                       model,
                       save_model=True,
                       model_basedir=os.path.join(os.path.expanduser('~'), '.keras'),
                       model_signature='p300model'):

    train_data, train_tags, test_data_with_noise, test_tags, noise_shifts = prepare_data_for_experiment(all_subjects,
                                add_time_domain_noise=add_time_domain_noise,
                                current_experiment_setting=current_experiment_setting,
                                downsample_params=downsample_params,
                                number_of_k_fold=number_of_k_fold,
                                cross_validation_iter=cross_validation_iter)

    callbacks=[]
    save_model = True
    if save_model:
        callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(model_basedir,"model_signature.hdf5"), monitor='val_loss', verbose=0, save_best_only=True,
                                        save_weights_only=False, mode='auto', period=1))

    model.fit(train_data.reshape(train_data.shape[0] * train_data.shape[1],
                                 train_data.shape[2], train_data.shape[3]), train_tags,
              verbose=1, epochs=30, batch_size=600, shuffle=True,
              validation_data=(test_data_with_noise[0].reshape(test_data_with_noise[0].shape[0] * test_data_with_noise[0].shape[1],
                                test_data_with_noise[0].shape[2], test_data_with_noise[0].shape[3]),test_tags),
              callbacks=callbacks)

    for time_shift_noise in noise_shifts:
        test_data = test_data_with_noise[time_shift_noise]
        accuracy_test, auc_score_test = predict_using_model(model,
                                                            test_data.reshape(
                                                                test_data.shape[0] * test_data.shape[1],
                                                                test_data.shape[2], test_data.shape[3]),
                                                            test_tags)
        print("cv:{} noise:{} accuracy_test {}, auc_score_train:{} ".format(cross_validation_iter,
                                                                               time_shift_noise,
                                                                               accuracy_test, auc_score_test))

    accuracy_train, auc_score_train = predict_using_model(model,
                                                          train_data.reshape(
                                                              train_data.shape[0] * train_data.shape[1],
                                                              train_data.shape[2], train_data.shape[3]),
                                                          train_tags)

    print("cv:{} accuracy_train {}, auc_score_train:{} ".format(cross_validation_iter, accuracy_train,
                                                                   auc_score_train))





def main():
    import keras
    keras.backend.set_image_dim_ordering('th')
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", help="name of the model of the experiment",
                        type=str, default="lstm_cnn_small")


    args = parser.parse_args()
    model_name = args.model_name
    # model_name = 'lstm_cnn_small'
    # model_name = 'LDA'
    # model_name = 'CNN'
    # model_name = 'lstm_small'
    # model_name = 'lstm_big'
    # model_name = 'lstm_cnn_big'
    downsample_params = 8

    current_experiment_setting = "Color116ms"

    all_subjects = [
        "gcd",
        "fat",
        "gcc",
        "icr",
        "icn",
        "iay",
        "gch",
        "gcg",
        "gcf",
        "gcb",
        "pia"
    ];

    add_time_domain_noise = False
    number_of_channels = 55

    number_of_k_fold = 10
    eeg_sample_shape = (int(200 / downsample_params), number_of_channels)

    for cross_validation_iter in range(number_of_k_fold):
        model = get_model(model_name, eeg_sample_shape)
        train_and_evaluate(all_subjects=all_subjects,
                           current_experiment_setting=current_experiment_setting,
                           downsample_params=downsample_params,
                           add_time_domain_noise=add_time_domain_noise,
                           cross_validation_iter=cross_validation_iter,
                           number_of_k_fold=number_of_k_fold,
                           model=model)




if __name__ == "__main__":
    main()



