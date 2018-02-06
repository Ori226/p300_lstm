import argparse
from utils import set_keras_backend
set_keras_backend("tensorflow")
from run_multi_subject_experiment import get_model, train_and_evaluate




def main():
    import keras
    keras.backend.set_image_dim_ordering('th')
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", help="name of the model of the experiment",
                        type=str, default="lstm_cnn_small")


    args = parser.parse_args()
    model_name = args.model_name
    model_name = 'lstm_cnn_small'
    # model_name = 'LDA'
    # model_name = 'CNN'
    # model_name = 'lstm_small'
    # model_name = 'lstm_big'
    # model_name = 'lstm_cnn_big'
    downsample_params = 8

    current_experiment_setting = "Color116ms"

    all_subjects = [
        "gcd",
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
        break




if __name__ == "__main__":
    main()



