import keras.models
from . import LEFT_OUT_MODEL_FOLDER
import os.path
import h5py
def load_left_out_model(model_file_name):
    model_full_path =  os.path.join(os.path.dirname(LEFT_OUT_MODEL_FOLDER), model_file_name)
    f = h5py.File(model_full_path, 'r+')
    if 'optimizer_weights' in f:
        del f['optimizer_weights']
    f.close()
    return keras.models.load_model(model_full_path)
