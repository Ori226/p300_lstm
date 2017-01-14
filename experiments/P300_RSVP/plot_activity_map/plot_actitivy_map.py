import mne

from mne.viz import plot_topomap


from OriKerasExtension.ThesisHelper import *
import keras


print "done"



def plot_weights_for_model(all_weights, subject_file_data):
    loc_xy = extract_2D_channel_location(subject_file_data)
    # print vals[0,0,0,:].shape

    for i, val in enumerate(all_weights):
        plt.subplot(3, 4, i)
        plot_topomap(val,np.fliplr(loc_xy),show=False)
    plt.show()
    print "done"


if __name__ == "__main__":
    subject_file_data = r'C:\Users\ORI\Documents\Thesis\dataset_all\RSVP_Color116msVPpia.mat'

    model_path = r"C:\git\thesis_clean_v3\experiments\P300_RSVP\model_left_out\model_smaller_lstm_cnn_all.h5"

    current_model = keras.models.load_model(model_path)
    all_weights = np.squeeze(current_model.layers[3].get_weights()[0])

    # all_weights = current_model.layers[3].get_weights()
    plot_weights_for_model(all_weights, subject_file_data)