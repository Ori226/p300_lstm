import keras
print keras.__version__

import numpy as np

from keras import backend as K
from keras.layers.core import Layer, RepeatVector
import numpy as np

class MyLayer2(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
    #     initial_weight_value = np.random.random((input_dim, self.output_dim))
    #     self.W = K.variable(initial_weight_value)
    #     self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x*0 +1 #.dot(x,x)*0 +1 +K.zeros((24,1)) #+ x.shape[0]
        # return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


            # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], self.output_dim)

def identity_loss(y_true, y_pred):

    # calculate the average loss over the whole batch:
    # K.reshape(dictionary_size*number_of_repetition, y_pred)
    # return K.mean(K.flatten(y_pred), axis=0)*0+ K.mean(y_pred,axis=1).shape[0]+0*y_pred.shape[0] - 0 * K.max(K.flatten(y_true))

    reshaped = K.mean(K.reshape(y_pred, (30, 3,-1)), axis=1)

    return K.mean(K.softmax(K.reshape(reshaped, (reshaped.shape[0], -1)))) +0*y_pred

    # return K.max(K.max(y_pred))*0+0*y_pred.shape[0] - 0 *K.max(K.max(K.flatten(y_true))) + 5
if __name__ == "__main__":

    # simple MLP model

    from keras.layers import  Dense
    from keras.models import  Sequential
    from keras.layers.recurrent import LSTM, GRU

    model = Sequential()
    model.add(

        LSTM(input_dim=55, output_dim=55, input_length=25, return_sequences=True))
    model.add(
        LSTM(input_dim=55, output_dim=30, input_length=25, return_sequences=False))
    # model.add(Dense(32, input_shape=(500,)))
    # model.add(Dense(32, input_shape=(500,)))

    # model.add(MyLayer2(6, input_shape=(1,)))
    model.compile(optimizer='rmsprop',
                  loss=identity_loss)

    # res = model.predict(np.random.rand(30*3*80,10,55).astype(np.float32))
    model.fit(np.random.rand(30*3*80,25,55).astype(np.float32), np.random.rand(30*3*80,1).astype(np.float32),batch_size=30*3*20)

    pass