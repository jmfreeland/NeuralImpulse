# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 17:51:22 2020
Object created to take an audio input file & audio output file and create a 
    neural network model to transform one to the other

@author: Josh Freeland
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
import pandas as pd

#set global variables for batch size, learning rate, epochs, optimizer, etc


class neuralTransform:
    
    def __init__(self, input_clip, output_clip, sample_rate, learning_rt, epoch_count, batch_sz, dropout_rate):
        #object requires an input file & an output file to start
        self.input_clip = librosa.load(input_clip, sr=sample_rate)
        self.output_clip = librosa.load(output_clip, sr=sample_rate)
        self.sample_rate = sample_rate
        self.learning_rt = learning_rt
        self.epoch_count = epoch_count
        self.batch_sz = batch_sz
        self.dropout_rate = dropout_rate

        #create different model types

        
    def plot_input(self):
        librosa.display.waveplot(y=self.input_clip[0], sr=self.sample_rate)
        
    def plot_output(self):
        librosa.display.waveplot(y=self.output_clip[0], sr=self.sample_rate)
    
    def fit_linear(self):
        
        #initialize new model
        self.linear_model = Sequential()
        #add single layer with one neuron
        self.linear_model.add(Dense(1, input_shape=(1,), input_dim=1))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                             beta_1=0.9, 
                                             beta_2=0.99, 
                                             epsilon=1e-05, 
                                             amsgrad=False,
                                             name='Adam')
        #compile linear model
        self.linear_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit linear model to data
        self.linear_history = self.linear_model.fit(self.input_clip[0], self.output_clip[0], epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)
        
    def transform_linear(self):
        return self.linear_model.predict(self.input_clip[0], use_multiprocessing=True, batch_size=self.batch_sz)
    
    def visualize_linear(self):
        print('creating linear_mode.png')
        plot_model(self.linear_model, to_file='linear_model.png', show_shapes=True, show_layer_names=True)
        
    def fit_linear_multi(self, lookback):
        #create new input and output lists including lookback
        self.linear_multi_model = Sequential()
        X, Y = [], []
        for i in range(len(self.input_clip[0]) - lookback):
            X.append(self.input_clip[0][i:i+lookback])
            Y.append(self.output_clip[0][i+lookback])
        #turn back into arrays
        self.multi_step_input = np.asarray(X)
        self.multi_step_output = np.asarray(Y)
        
        del X, Y
        #add a multi-input basic layer
        self.linear_multi_model.add(Dense(1, input_shape=(lookback,), input_dim=lookback))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                              beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='SGD')
        #compile linear model
        self.linear_multi_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit linear model to data
        self.linear_multi_history = self.linear_multi_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)


    def transform_linear_multi(self):
        return self.linear_multi_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)


    def fit_dense(self, neurons):
        #create dense model with multiple neurons
        self.dense_model = Sequential(name='dense')
        
        #add a multi-input basic layer with parameterized neurons
        self.dense_model.add(Dense(neurons, name='dense_1', activation='tanh', input_shape=(self.multi_step_input.shape[1],), input_dim=self.multi_step_input.shape[1]))
        self.dense_model.add(Dropout(rate=self.dropout_rate))
        self.dense_model.add(BatchNormalization())
        self.dense_model.add(Dense(1, name='dense_out'))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                              beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='SGD')
        #compile dense single-layer model
        self.dense_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit model to data
        self.dense_history = self.dense_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)

    def transform_dense(self):
        return self.dense_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)

    
    def fit_dense2(self, neurons_l1, neurons_l2):
        #lets initialize a another model now and add two layers
        self.dense2_model = Sequential(name='dense2')
        
        #add a multi-input basic layer and second layer
        self.dense2_model.add(Dense(neurons_l1, name='dense2_1', activation='tanh', input_shape=(self.multi_step_input.shape[1],), input_dim=self.multi_step_input.shape[1]))
        self.dense2_model.add(Dropout(rate=self.dropout_rate))
        self.dense2_model.add(BatchNormalization())
        self.dense2_model.add(Dense(neurons_l2, name='dense2_2', activation='tanh'))
        self.dense2_model.add(Dense(1, name='dense2_out'))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                              beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='Adam')
        #compile two-layer model
        self.dense2_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit model to data
        self.dense2_history = self.dense2_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)
        
    def transform_dense2(self):
        return self.dense2_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)

    def fit_dense3(self, neurons_l1, neurons_l2, neurons_l3):
        #lets initialize a another model now and add two layers
        self.dense3_model = Sequential(name='dense3')
        
        #add a multi-input basic layer and second layer
        self.dense3_model.add(Dense(neurons_l1, name='dense3_1', activation='tanh', input_shape=(self.multi_step_input.shape[1],), input_dim=self.multi_step_input.shape[1]))
        self.dense3_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_model.add(BatchNormalization())
        self.dense3_model.add(Dense(neurons_l2, name='dense3_2', activation='tanh'))
        self.dense3_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_model.add(BatchNormalization())
        self.dense3_model.add(Dense(neurons_l3, name='dense3_3', activation='tanh'))
        self.dense3_model.add(Dense(1, name='dense3_out'))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                              beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='Adam')
        #compile two-layer model
        self.dense3_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit model to data
        self.dense3_history = self.dense3_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)
        
    def transform_dense3(self):
        return self.dense3_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)

    def fit_rnn_dense3(self, neurons_l1, neurons_prernn1, neurons_l2, neurons_l3):
        #lets initialize a another model now and add two layers
        self.rnn_dense3_model = Sequential(name='rnn_dense3')
        
        #add a multi-input basic layer and second layer
        self.rnn_dense3_model.add(Dense(neurons_l1, name='rnn_dense3_1', activation='tanh', input_shape=(self.multi_step_input.shape[1],), input_dim=self.multi_step_input.shape[1]))
        self.rnn_dense3_model.add(tf.keras.layers.Reshape((neurons_l1, 1)))
        self.rnn_dense3_model.add(Dropout(rate=self.dropout_rate))
        self.rnn_dense3_model.add(BatchNormalization())
        self.rnn_dense3_model.add(SimpleRNN(neurons_prernn1, name='rnn_dense3_r1', activation='tanh'))
        self.rnn_dense3_model.add(Dropout(rate=self.dropout_rate))
        self.rnn_dense3_model.add(BatchNormalization())
        self.rnn_dense3_model.add(Dense(neurons_l2, name='rnn_dense3_2', activation='tanh'))
        self.rnn_dense3_model.add(Dropout(rate=self.dropout_rate))
        self.rnn_dense3_model.add(BatchNormalization())
        self.rnn_dense3_model.add(Dense(neurons_l3, name='rnn_dense3_3', activation='tanh'))
        self.rnn_dense3_model.add(Dense(1, name='rnn_dense3_out'))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                              beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='Adam')
        #compile two-layer model
        self.rnn_dense3_model.compile(loss='mae', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit model to data
        self.rnn_dense3_history = self.rnn_dense3_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)
        
    def transform_rnn_dense3(self):
        return self.rnn_dense3_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)


    def fit_dense3_rnn(self, neurons_l1, neurons_l2, neurons_l3):

        self.dense3_rnn_model = Sequential(name='dense3_rnn')
                
                #add a multi-input basic layer and second layer
        self.dense3_rnn_model.add(Dense(neurons_l1, name='dense3_rnn_1', kernel_initializer='glorot_normal', activation='tanh', input_shape=(self.multi_step_input.shape[1],), input_dim=self.multi_step_input.shape[1]))
        self.dense3_rnn_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_rnn_model.add(BatchNormalization())
        self.dense3_rnn_model.add(Dense(neurons_l2, name='dense3_rnn_2', kernel_initializer='glorot_normal', activation='tanh'))
        self.dense3_rnn_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_rnn_model.add(BatchNormalization())
        self.dense3_rnn_model.add(Dense(neurons_l3, name='dense3_rnn_3', kernel_initializer='glorot_normal', activation='tanh'))
        self.dense3_rnn_model.add(tf.keras.layers.Reshape((neurons_l3, 1)))
        self.dense3_rnn_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_rnn_model.add(BatchNormalization())
        self.dense3_rnn_model.add(SimpleRNN(1, name='dense3_rnn_out', activation='tanh'))
                #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                                     beta_1=0.9, 
                                                      beta_2=0.99, 
                                                      epsilon=1e-05, 
                                                      amsgrad=False,
                                                      name='Adam')
                #compile two-layer model
        self.dense3_rnn_model.compile(loss='mse', 
                                          optimizer= optimizer, 
                                          metrics=['mse', 'mae'])
        
        self.dense3_rnn_model.summary()
        self.dense3_rnn_history = self.dense3_rnn_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)
        
    def transform_dense3_rnn(self):
        return self.dense3_rnn_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)

    def fit_lstm_dense3(self, neurons_l1, neurons_prernn1, neurons_l2, neurons_l3):
        #lets initialize a another model now and add two layers
        self.lstm_dense3_model = Sequential(name='lstm_dense3')
        
        #add a multi-input basic layer and second layer
        self.lstm_dense3_model.add(Dense(neurons_l1, name='lstm_dense3_1', activation='tanh', input_shape=(self.multi_step_input.shape[1],), input_dim=self.multi_step_input.shape[1]))
        self.lstm_dense3_model.add(tf.keras.layers.Reshape((neurons_l1, 1)))
        self.lstm_dense3_model.add(Dropout(rate=self.dropout_rate))
        self.lstm_dense3_model.add(BatchNormalization())
        self.lstm_dense3_model.add(LSTM(neurons_prernn1, name='lstm_dense3_r1', activation='tanh'))
        self.lstm_dense3_model.add(Dropout(rate=self.dropout_rate))
        self.lstm_dense3_model.add(BatchNormalization())
        self.lstm_dense3_model.add(Dense(neurons_l2, name='lstm_dense3_2', activation='tanh'))
        self.lstm_dense3_model.add(Dropout(rate=self.dropout_rate))
        self.lstm_dense3_model.add(BatchNormalization())
        self.lstm_dense3_model.add(Dense(neurons_l3, name='lstlm_dense3_3', activation='tanh'))
        self.lstm_dense3_model.add(Dense(1, name='lstm_dense3_out'))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                              beta_1=0.9, 
                                              beta_2=0.99, 
                                              epsilon=1e-05, 
                                              amsgrad=False,
                                              name='Adam')
        #compile two-layer model
        self.lstm_dense3_model.compile(loss='mae', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit model to data
        self.lstm_dense3_history = self.lstm_dense3_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)

    def transform_lstm_dense3(self):
        return self.lstm_dense3_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)

    def fit_dense3_lstm(self, neurons_l1, neurons_l2, neurons_l3):

        self.dense3_lstm_model = Sequential(name='dense3_lstm')
                
                #add a multi-input basic layer and second layer
        self.dense3_lstm_model.add(Dense(neurons_l1, name='dense3_lstm_1', kernel_initializer='glorot_normal', activation='tanh', input_shape=(self.multi_step_input.shape[1],), input_dim=self.multi_step_input.shape[1]))
        self.dense3_lstm_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_lstm_model.add(BatchNormalization())
        self.dense3_lstm_model.add(Dense(neurons_l2, name='dense3_lstm_2', kernel_initializer='glorot_normal', activation='tanh'))
        self.dense3_lstm_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_lstm_model.add(BatchNormalization())
        self.dense3_lstm_model.add(Dense(neurons_l3, name='dense3_lstm_3', kernel_initializer='glorot_normal', activation='tanh'))
        self.dense3_lstm_model.add(tf.keras.layers.Reshape((neurons_l3, 1)))
        self.dense3_lstm_model.add(Dropout(rate=self.dropout_rate))
        self.dense3_lstm_model.add(BatchNormalization())
        self.dense3_lstm_model.add(LSTM(1, name='dense3_lstm_out', activation='tanh'))
                #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rt, 
                                                     beta_1=0.9, 
                                                      beta_2=0.99, 
                                                      epsilon=1e-05, 
                                                      amsgrad=False,
                                                      name='Adam')
                #compile two-layer model
        self.dense3_lstm_model.compile(loss='mse', 
                                          optimizer= optimizer, 
                                          metrics=['mse', 'mae'])
        
        self.dense3_lstm_model.summary()
        self.dense3_lstm_history = self.dense3_lstm_model.fit(self.multi_step_input, self.multi_step_output, epochs=self.epoch_count, batch_size=self.batch_sz, use_multiprocessing=True, workers=8)
        
    def transform_dense3_lstm(self):
        return self.dense3_lstm_model.predict(self.multi_step_input, use_multiprocessing=True, batch_size=self.batch_sz)


    #-----Output functions
    def plot_errors(self):
        mse_series = pd.Series()
        mse_series['linear'] = self.linear_history.history['mse'][-1]
        mse_series['linear_multi'] = self.linear_multi_history.history['mse'][-1]
        mse_series['dense'] = self.dense_history.history['mse'][-1]
        mse_series['dense2'] = self.dense2_history.history['mse'][-1]
        mse_series['dense3'] = self.dense3_history.history['mse'][-1]
        mse_series['rnn_dense3'] = self.rnn_dense3_history.history['mse'][-1]
        mse_series['dense3_rnn'] = self.dense3_rnn_history.history['mse'][-1]     
        mse_series['lstm_dense3'] = self.lstm_dense3_history.history['mse'][-1]    
        mse_series['dense3_lstm'] = self.dense3_lstm_history.history['mse'][-1]    
        self.mse_series = mse_series
        fig = plt.figure(figsize=(8,5), dpi=300)
        sns.barplot(x=mse_series.index, y=mse_series.values)
        fig.suptitle('Mean Squared Error by Model')
        
        mae_series = pd.Series()
        mae_series['linear'] = self.linear_history.history['mae'][-1]
        mae_series['linear_multi'] = self.linear_multi_history.history['mae'][-1]
        mae_series['dense'] = self.dense_history.history['mae'][-1]
        mae_series['dense2'] = self.dense2_history.history['mae'][-1]
        mae_series['dense3'] = self.dense3_history.history['mae'][-1]
        mae_series['rnn_dense3'] = self.rnn_dense3_history.history['mae'][-1]
        mae_series['dense3_rnn'] = self.dense3_rnn_history.history['mae'][-1]
        mae_series['lstm_dense3'] = self.lstm_dense3_history.history['mae'][-1]    
        mae_series['dense3_lstm'] = self.dense3_lstm_history.history['mae'][-1]    
        self.mae_series = mae_series
        fig = plt.figure(figsize=(8,5), dpi=300)
        sns.barplot(x=mse_series.index, y=mae_series.values)
        fig.suptitle('Mean Absolute Error by Model')
        

    
    