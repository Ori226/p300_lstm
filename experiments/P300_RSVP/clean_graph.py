# from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten
# from keras.models import Model
#
# # first, define the vision modules
# digit_input = Input(shape=(1, 27, 27))
# x = Convolution2D(64, 3, 3)(digit_input)
# x = Convolution2D(64, 3, 3)(x)
# x = MaxPooling2D((2, 2))(x)
# out = Flatten()(x)
#
# vision_model = Model(digit_input, out)
#
# # then define the tell-digits-apart model
# digit_a = Input(shape=(1, 27, 27))
# digit_b = Input(shape=(1, 27, 27))
#
# # the vision model will be shared, weights and all
# out_a = vision_model(digit_a)
# out_b = vision_model(digit_b)
#
# concatenated = merge([out_a, out_b], mode='concat')
# out = Dense(1, activation='sigmoid')(concatenated)
#
# classification_model = Model([digit_a, digit_b], out)



from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, Lambda
from keras.models import Model

# first, define the vision modules
from keras import backend as K

def per_char_loss(X):
    #alls = X.values()
    concatenated = X #K.concatenate(alls)
    reshaped = K.mean(K.reshape(concatenated, (K.shape(concatenated)[0], 3, 30)), axis=1)

    return reshaped#K.softmax(K.reshape(reshaped, (reshaped.shape[0], -1)))

# eeg_sample_size = (25,55)
eeg_sample_shape = (200,55)
digit_input = Input(shape=eeg_sample_shape)
x = Flatten(input_shape=(200, 55))(digit_input)
x = Dense(40)(x)
x = Activation('tanh')(x)
x = Dropout(0.1)(x)
x = Dense(40)(x)
x = Dropout(0.1)(x)
x = Activation('tanh')(x)
x = Dense(1)(x)
out = x #Flatten()(x)

# model = Sequential()
#     model.add(Flatten(input_shape=(number_of_time_stamps, number_of_in_channels)))
#     model.add(Dense(40))
#     model.add(Dropout(0.1))
#     model.add(Activation('tanh'))
#     model.add(Dense(40))
#     model.add(Dropout(0.1))
#     model.add(Activation('tanh'))
#     model.add(Dense(1))




# P300 identification model
p300_identification_model = Model(digit_input, out)

all_inputs = [ Input(shape=eeg_sample_shape) for _ in range(90)]

all_outs = [ p300_identification_model(input_i) for input_i in all_inputs]

concatenated = merge(all_outs, mode='concat')
out = Lambda(per_char_loss, output_shape=(30,))(concatenated)
# out = Dense(30)(concatenated)


# out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model(all_inputs, out)
classification_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# loss='categorical_crossentropy

import numpy as np
print classification_model.predict([np.random.rand(3,200,55).astype(np.float32)for _ in range(90) ]).shape


# get the weight of the model:
print p300_identification_model.get_weights()


history = classification_model.fit([np.random.rand(3,200,55).astype(np.float32)for _ in range(90) ], np.random.rand(3, 30).astype(np.float32), nb_epoch=1)
print p300_identification_model.get_weights()

print history.history