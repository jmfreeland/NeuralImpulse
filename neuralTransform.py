# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 17:51:22 2020
Object created to take an audio input file & audio output file and create a 
    neural network model to transform one to the other

@author: Josh Freeland
"""

import librosa
import librosa.display
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np

class neuralTransform:
    
    def __init__(self, input_clip, output_clip, sample_rate):
        #object requires an input file & an output file to start
        self.input_clip = librosa.load(input_clip, sr=sample_rate)
        self.output_clip = librosa.load(output_clip, sr=sample_rate)
        self.sample_rate = sample_rate

        #create different model types
        self.linear_model = Sequential()
        self.linear_multi_model = Sequential()
        
        
        
    def plot_input(self):
        librosa.display.waveplot(y=self.input_clip[0], sr=self.sample_rate)
        
    def plot_output(self):
        librosa.display.waveplot(y=self.output_clip[0], sr=self.sample_rate)
    
    def fit_linear(self):
        #add single layer with one neuron
        self.linear_model.add(Dense(1, input_shape=(1,), input_dim=1))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, 
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
        self.linear_model.fit(self.input_clip[0], self.output_clip[0], epochs=4, batch_size=1024, use_multiprocessing=True, workers=4)
        
    def transform_linear(self):
        return self.linear_model.predict(self.input_clip[0], use_multiprocessing=True, batch_size=1024)
    
    def visualize_linear(self):
        print('creating linear_mode.png')
        plot_model(self.linear_model, to_file='linear_model.png', show_shapes=True, show_layer_names=True)
        
    def transform_linear_multi(self, lookback):
        #create new input and output lists including lookback
        X, Y = [], []
        for i in range(len(self.input_clip[0]) - lookback):
            X.append(self.input_clip[0][i:i+lookback])
            Y.append(self.output_clip[0][i+lookback])
        #turn back into arrays
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        #add a multi-input basic layer
        self.linear_multi_model.add(Dense(lookback, input_shape=(lookback,), input_dim=lookback))
        #create optimizer (to be refined)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, 
                                             beta_1=0.9, 
                                             beta_2=0.99, 
                                             epsilon=1e-05, 
                                             amsgrad=False,
                                             name='Adam')
        #compile linear model
        self.linear_multi_model.compile(loss='mse', 
                                  optimizer= optimizer, 
                                  metrics=['mse', 'mae'])
        #fit linear model to data
        self.linear_multi_model.fit(X, Y, epochs=4, batch_size=1024, use_multiprocessing=True, workers=4)
    