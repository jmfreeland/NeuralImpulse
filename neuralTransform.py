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


class neuralTransform:
    
    def __init__(self, input_clip, output_clip, sample_rate):
        #object requires an input file & an output file to start
        self.input_clip = librosa.load(input_clip, sr=sample_rate)
        self.output_clip = librosa.load(output_clip, sr=sample_rate)
        self.sample_rate = sample_rate

        #create different model types
        self.linear_model = Sequential()
        
        
        
    def plot_input(self):
        librosa.display.waveplot(y=self.input_clip[0], sr=self.sample_rate)
        
    def plot_output(self):
        librosa.display.waveplot(y=self.output_clip[0], sr=self.sample_rate)
    
    def fit_linear(self):
        self.linear_model.add(Dense(1,input_dim=1))
        self.linear_model.compile(loss='binary_crossentropy', 
                                  optimizer='adam', 
                                  metrics=['accuracy'])
        self.linear_model.fit(self.input_clip[0], self.output_clip[0], epochs=10, batch_size=1000)
        
    def transform_linear(self):
        return self.linear_model.predict(self.input_clip[0])
    
    def visualize_linear(self):
        print('creating linear_mode.png')
        plot_model(self.linear_model, to_file='linear_model.png', show_shapes=True, show_layer_names=True)