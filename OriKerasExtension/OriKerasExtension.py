import keras
print('Hello World4')

import theano
import matplotlib.pyplot as plt
from keras import activations, initializations
# from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from keras.layers.core import Layer, MaskedLayer
from keras.layers.recurrent import Recurrent,LSTM
from theano import tensor as T
# from six.moves import range
import numpy as np


#
# class DebugLSTM(LSTM):
#
#     def _step2(self,
#               xi_t, xf_t, xo_t, xc_t, mask_tm1,
#               h_tm1, c_tm1,
#               u_i, u_f, u_o, u_c):
#         h_mask_tm1 = mask_tm1 * h_tm1
#         c_mask_tm1 = mask_tm1 * c_tm1
#
#         i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
#         f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
#         c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
#         o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
#         h_t = o_t * self.activation(c_t)
#         return h_t, c_t, i_t, f_t, o_t
#
#     def get_output2(self, train=False):
#         X = self.get_input(train)
#         padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
#         X = X.dimshuffle((1, 0, 2))
#
#         xi = T.dot(X, self.W_i) + self.b_i
#         xf = T.dot(X, self.W_f) + self.b_f
#         xc = T.dot(X, self.W_c) + self.b_c
#         xo = T.dot(X, self.W_o) + self.b_o
#
#         [outputs, memories, write_g, forget_g, read_g], updates = theano.scan(
#             self._step2,
#             sequences=[xi, xf, xo, xc, padded_mask],
#             outputs_info=[
#                 T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                 T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                 None,
#                 None,
#                 None
#                 ],
#             non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
#             truncate_gradient=self.truncate_gradient)
#
#         return  outputs.dimshuffle((1, 0, 2)), memories.dimshuffle((1, 0, 2)), \
#                 write_g.dimshuffle((1, 0, 2)), \
#                 forget_g.dimshuffle((1, 0, 2)), \
#                 read_g.dimshuffle((1, 0, 2)),\
#                 X.dimshuffle((1, 0, 2))
#

def plotLSTMExample(prediction, example_index):    
    plt.figure(figsize=(40,10))
    plt.subplot(1, 7, 1)
    plt.title('input')
    plt.imshow(np.asarray(prediction[5])[example_index,:,:],  aspect='auto', interpolation='none')
    # plt.show()
    
    
    plt.subplot(1, 7, 2)
    plt.title('activation')
    plt.imshow(np.asarray(prediction[0])[example_index,:,:],  aspect='auto', interpolation='none')
    # plt.show()

    
    plt.subplot(1, 7, 3)
    plt.title('memory')
    plt.imshow(np.asarray(prediction[1])[example_index,:,:],  aspect='auto', interpolation='none')
    # plt.show()

    
    plt.subplot(1, 7, 4)
    plt.title('write')
    plt.imshow(np.asarray(prediction[2])[example_index,:,:],  aspect='auto', interpolation='none')
    # plt.show()

    # print ('forget')
    plt.subplot(1, 7, 5,aspect='auto')
    plt.title('forget')
    plt.imshow(np.asarray(prediction[3])[example_index,:,:],  aspect='auto', interpolation='none')
    # plt.show()
    plt.subplot(1, 7, 6)
    plt.title('read')
    plt.imshow(np.asarray(prediction[4])[example_index,:,:],  aspect='auto', interpolation='none')
    plt.show()


def foo():
    print('dsdsd')
