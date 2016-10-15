from keras.layers import Reshape
import keras.backend as K
if __name__ == "__main__":
    margin_alpha = 0.3

    def get_batch(X_train,y_train):
        data = np.expand_dims(X_train, 1)
        tags = y_train
        while True:
            yield [[data[0:2], data[2:4], data[4:6]], tags[0:2]]

    def triplet_loss(X):
        pos_distnace = K.sum(K.square(X[0] - X[1]),axis=1)
        neg_distnace = K.sum(K.square(X[0] - X[2]), axis=1)
        # return K.l2_normalize(X[0] - X[1], axis=0)*0 - K.l2_normalize(X[0]- X[2], axis=0)*0 + margin_alpha
        return K.maximum(pos_distnace - neg_distnace + margin_alpha,0)


    def triplet_loss_shape(X):
        return (X[0][0],1)

    def get_triplet_model(input_shape):
        import keras.backend as K
        from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Lambda
        from keras.models import Model


        # first, define the vision modules
        digit_input = Input(shape=input_shape)
        x = Convolution2D(64, 3, 3)(digit_input)
        x = Convolution2D(64, 3, 3)(x)
        x = MaxPooling2D((2, 2))(x)
        out = Flatten()(x)

        P300_model = Model(digit_input, out)

        # then define the tell-digits-apart model
        input_anchor = Input(shape=input_shape)
        input_positive = Input(shape=input_shape)
        input_negative = Input(shape=input_shape)

        out_anchor = P300_model(input_anchor)
        out_positive = P300_model(input_positive)
        out_negative = P300_model(input_negative)


        out = merge([out_anchor, out_positive, out_negative], mode=triplet_loss,output_shape=triplet_loss_shape)

        # out = Dense(1, activation='sigmoid')(concatenated)


        classification_model = Model([input_anchor, input_positive, input_negative], out)
        return classification_model



    def identity_loss(y_true, y_pred):

        return K.mean(y_pred + 0 * K.sum(y_true) )
    def antirectifier(x):
        import keras.backend as K
        x -= K.mean(x, axis=1, keepdims=True)
        x = K.l2_normalize(x, axis=1)
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)


    def antirectifier_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)


    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    sample_shape = (1,28,28)

    triplet_model = get_triplet_model(sample_shape)
    triplet_model.compile(optimizer='rmsprop', loss=identity_loss)
    import numpy as np
    temp = np.expand_dims(X_train, 1)
    # print triplet_model.predict([temp[0:2], temp[0:2], temp[0:2]])

    batch_generator = get_batch(X_train,y_train=y_train)

    triplet_model.fit_generator(batch_generator,nb_epoch=4,samples_per_epoch=2)
    print triplet_model.summary()

    # building the batches:


    print "here"

    # model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
    #
    # # the vision model will be shared, weights and all
    # out_anchor = P300_model(input_anchor)
    # out_positive = P300_model(input_positive)
    # out_negative = P300_model(input_negative)
    #
    # concatenated = merge([out_anchor, out_positive, out_negative], mode='concat')
    # out = Dense(1, activation='sigmoid')(concatenated)
    #
    # classification_model = Model([input_anchor, input_positive], out)
    pass