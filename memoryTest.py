# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:34:37 2020

@author: freel
"""
import librosa
import librosa.display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

lookback=600
X, Y = [], []
for i in range(len(test_audio.input_clip[0]) - lookback):
    X.append(test_audio.input_clip[0][i:i+lookback])
    Y.append(test_audio.output_clip[0][i+lookback])
        #turn back into arrays
multi_step_input = np.asarray(X)
multi_step_output = np.asarray(Y)



lookback=1000
X, Y = np.array([[]]), np.array([[]])
for i in range(len(test_audio.input_clip[0]) - lookback):
    X = np.append(X, test_audio.input_clip[0][i:i+lookback])
    Y = np.append(Y, test_audio.output_clip[0][i+lookback])
        #turn back into arrays        
        

neurons_l1 = 128
neurons_l2 = 64
neurons_l3 = 32
neurons_prernn1 = 128


#build rnn->dense model        
rnn_dense3_model = Sequential(name='rnn_dense3')
        
        #add a multi-input basic layer and second layer
rnn_dense3_model.add(Dense(neurons_l1, name='rnn_dense3_1', activation='tanh', input_shape=(test_audio.multi_step_input.shape[1],), input_dim=test_audio.multi_step_input.shape[1]))
rnn_dense3_model.add(tf.keras.layers.Reshape((neurons_l1, 1)))
rnn_dense3_model.add(SimpleRNN(10, name='rnn_dense3_r1', activation='tanh'))
rnn_dense3_model.add(Dense(neurons_l2, name='rnn_dense3_2', activation='tanh'))
rnn_dense3_model.add(Dense(neurons_l2, name='rnn_dense3_3', activation='tanh'))
rnn_dense3_model.add(Dense(1, name='rnn_dense3_out'))
        #create optimizer (to be refined)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, 
                                             beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='Adam')
        #compile two-layer model
rnn_dense3_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])

rnn_dense3_model.summary()


#build dense->rnn model
neurons_l1 = 128
neurons_l2 = 64
neurons_l3 = 32
epoch_count = 2
batch_sz = 4096

dense3_rnn_model = Sequential(name='dense3_rnn')
        
        #add a multi-input basic layer and second layer
dense3_rnn_model.add(Dense(neurons_l1, name='rnn_dense3_1', kernel_initializer='glorot_normal', activation='tanh', input_shape=(test_audio.multi_step_input.shape[1],), input_dim=test_audio.multi_step_input.shape[1]))
dense3_rnn_model.add(Dense(neurons_l2, name='rnn_dense3_2', kernel_initializer='glorot_normal', activation='tanh'))
dense3_rnn_model.add(Dense(neurons_l3, name='rnn_dense3_3', kernel_initializer='glorot_normal', activation='tanh'))
dense3_rnn_model.add(tf.keras.layers.Reshape((neurons_l3, 1)))
dense3_rnn_model.add(SimpleRNN(1, name='rnn_dense3_out', activation='tanh'))
        #create optimizer (to be refined)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, 
                                             beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='Adam')
        #compile two-layer model
dense3_rnn_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])

dense3_rnn_model.summary()
dense3_rnn_model.fit(test_audio.multi_step_input, test_audio.multi_step_output, epochs=epoch_count, batch_size=batch_sz, use_multiprocessing=True, workers=5)
