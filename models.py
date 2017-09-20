# lstm
# cnn-lstm
# cnn
# lda
from abc import ABCMeta, abstractmethod

from keras.engine import Input, Model
from keras.layers import Convolution2D, Activation, Permute, LSTM, Dense, Flatten
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class GeneralModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, _X, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, _X, _y, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


def get_only_P300_model_LSTM_CNN(eeg_sample_shape, number_of_hidden=30):
    digit_input = Input(shape=eeg_sample_shape)
    from keras.layers.core import Reshape
    x = Reshape((1, eeg_sample_shape[0], eeg_sample_shape[1]))(digit_input)
    x = Convolution2D(nb_filter=10,
                      nb_col=eeg_sample_shape[1],
                      nb_row=1,
                      border_mode='valid',
                      init='glorot_uniform')(x)
    x = Activation('tanh')(x)
    x = Permute((3, 2, 1))(x)
    x = Reshape((eeg_sample_shape[0], 10))(x)
    x = LSTM(number_of_hidden, return_sequences=False, consume_less='mem')(x)
    x = Dense(1)(x)
    out = Activation(activation='sigmoid')(x)

    model = Model(digit_input, out)
    model.summary()
    return model

def get_only_P300_model_LSTM(eeg_sample_shape, number_of_hidden=30):
    from keras.regularizers import l2
    input_layer = Input(shape=eeg_sample_shape)
    # x = Flatten(input_shape=eeg_sample_shape)(digit_input)
    # x = noise.GaussianNoise(sigma=0.0)(digit_input)
    x = LSTM(number_of_hidden,input_shape=eeg_sample_shape,return_sequences=False)(input_layer)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(input_layer, out)
    return model

def get_only_P300_model_CNN(eeg_sample_shape):
    digit_input = Input(shape=eeg_sample_shape)
    from keras.layers.core import Reshape

    x = Reshape((1, eeg_sample_shape[0], eeg_sample_shape[1]))(digit_input)
    x = Convolution2D(nb_filter=10,
                                 nb_col=eeg_sample_shape[1],
                                 nb_row=1,
                                 border_mode='valid',
                                 init='glorot_uniform')(x)
    x= Activation('tanh')(x)
    x = Convolution2D(nb_filter=13,
                      nb_col=1,
                      nb_row=5,
                      subsample=(5,1),
                      border_mode='valid',
                      init='glorot_uniform')(x)
    x = Activation('tanh')(x)
    x = Flatten()(x)
    x = Dense(100, )(x)
    x = Activation('sigmoid')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    model = Model(digit_input, x)
    return model

class My_LDA_public(LDA, GeneralModel):
    def reset(self):
        super(My_LDA_public, self).reset()

    def predict(self, _X, *args, **kwargs):
        prediction_results =super(My_LDA_public, self).predict_proba(_X.reshape(_X.shape[0], -1))

        return prediction_results[:,1]

    def fit(self, _X, _y, *args, **kwargs):
        return super(My_LDA_public, self).fit(_X.reshape(_X.shape[0], -1), _y.flatten())

    def get_name(self):
        super(My_LDA_public, self).get_name()
        return self.__class__.__name__

    def get_params(self):
        return None