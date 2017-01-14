import mne

from  mne.viz import plot_topomap


from  OriKerasExtension.ThesisHelper import *

model_path = r"C:\git\thesis_clean_v3\experiments\P300_RSVP\model_left_out\RSVP_Color116msVPfat.h5"
import keras
current_model = keras.models.load_model(model_path)
all_weights = np.squeeze(current_model.layers[3].get_weights()[0])

loc_xy = extract_2D_channel_location(r'C:\Users\ORI\Documents\Thesis\dataset_all\RSVP_Color116msVPpia.mat')

current_model.layers[3].get_weights()

print "done"

# print vals[0,0,0,:].shape
for val in all_weights:
    plot_topomap(val,np.fliplr(loc_xy),show=True)
    plt.show()
    print "done"