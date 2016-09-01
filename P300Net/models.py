from keras import backend as K
from keras.layers import Dense, Lambda
from keras.models import Sequential, Graph


def get_item_subgraph(input_shape, latent_dim):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout

    number_of_time_stamps = 200
    number_of_in_channels = 55
    model = Sequential()
    model.add(Flatten(input_shape=(number_of_time_stamps, number_of_in_channels)))
    model.add(Dense(40))
    model.add(Dropout(0.1))
    model.add(Activation('tanh'))
    model.add(Dense(40))
    model.add(Dropout(0.1))
    model.add(Activation('tanh'))
    model.add(Dense(1))

    return model

def get_item_lstm_subgraph(number_of_timestamps, number_of_channels):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.recurrent import LSTM, GRU
    from keras.regularizers import l2

    _num_of_hidden_units = 100
    model = Sequential()
    # model.add(Flatten(input_shape=(number_of_timestamps, number_of_channels)))
    # model.add(Dense(40))
    model.add(LSTM(input_dim=55, output_dim=_num_of_hidden_units, input_length=number_of_timestamps, return_sequences=True))
    print "shape:{}".format(model.layers[-1].output_shape)
    # model.add(Dropout(0.3))

    model.add(LSTM(input_dim=_num_of_hidden_units, output_dim=_num_of_hidden_units, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_user_subgraph(input_shape, latent_dim):
    model = Sequential()
    model.add(Dense(latent_dim, input_shape=input_shape))
    return model


def per_char_loss(X):
    alls = X.values()
    concatenated = K.concatenate(alls)
    reshaped = K.mean(K.reshape(concatenated, (K.shape(concatenated)[0], 3, 30)), axis=1)

    return K.softmax(K.reshape(reshaped, (reshaped.shape[0], -1)))


def identity_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)  # K.mean(y_pred - 0 * y_true)


def get_graph(num_items, latent_dim, number_of_timestamps, number_of_channels):
    batch_input_shape = (1, number_of_timestamps, number_of_channels)
    # batch_input_shape = None
    magic_num = 90
    model = Graph()


    for i in range(magic_num):
        model.add_input('positive_item_input_{}'.format(i), input_shape=(number_of_timestamps, number_of_channels), batch_input_shape=batch_input_shape)

    model.add_shared_node(get_item_subgraph((num_items,), latent_dim),
                          name='item_latent',
                          inputs=["positive_item_input_{}".format(i) for i in range(magic_num)],
                          merge_mode='join')

    # Compute loss
    model.add_node(Lambda(per_char_loss),
                   name='triplet_loss',
                   input='item_latent')

    # Add output
    model.add_output(name='triplet_loss', input='triplet_loss')
    model.compile(loss={'triplet_loss': identity_loss}, optimizer='sgd')  # Adagrad(lr=0.1, epsilon=1e-06))

    return model


def get_graph_lstm(num_items, latent_dim, number_of_timestamps, number_of_channels):
    batch_input_shape = (1, number_of_timestamps, number_of_channels)
    # batch_input_shape = None
    magic_num = 90
    model = Graph()


    for i in range(magic_num):
        model.add_input('positive_item_input_{}'.format(i), batch_input_shape=batch_input_shape)

    model.add_shared_node(get_item_lstm_subgraph(number_of_timestamps,number_of_channels),
                          name='item_latent',
                          inputs=["positive_item_input_{}".format(i) for i in range(magic_num)],
                          merge_mode='join')

    # Compute loss
    model.add_node(Lambda(per_char_loss),
                   name='triplet_loss',
                   input='item_latent')

    # Add output
    model.add_output(name='triplet_loss', input='triplet_loss')
    model.compile(loss={'triplet_loss': identity_loss}, optimizer='rmsprop')  # Adagrad(lr=0.1, epsilon=1e-06))

    return model



def get_graph_lstm_new_keras(num_items, latent_dim, number_of_timestamps, number_of_channels):
    batch_input_shape = (1, number_of_timestamps, number_of_channels)
    # batch_input_shape = None
    magic_num = 90
    model = Graph()


    for i in range(magic_num):
        model.add_input('positive_item_input_{}'.format(i), batch_input_shape=batch_input_shape)

    model.add_shared_node(get_item_lstm_subgraph(number_of_timestamps,number_of_channels),
                          name='item_latent',
                          inputs=["positive_item_input_{}".format(i) for i in range(magic_num)],
                          merge_mode='concat')

    # Compute loss
    model.add_node(Lambda(per_char_loss),
                   name='triplet_loss',
                   input='item_latent')

    # Add output
    model.add_output(name='triplet_loss', input='triplet_loss')
    model.compile(loss={'triplet_loss': identity_loss}, optimizer='rmsprop')  # Adagrad(lr=0.1, epsilon=1e-06))

    return model