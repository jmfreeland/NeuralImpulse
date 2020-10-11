# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:00:25 2020

Script to test basic transform model

@author: freel
"""
import neuralTransform as nt
import librosa
import librosa.display
import matplotlib.pyplot as plt


test_audio = nt.neuralTransform('C:/users/freel/Desktop/neuralImpulse/SL68Trial_Raw.aif',
                            'C:/users/freel/Desktop/neuralImpulse/SL68Trial.aif',
                            48000)

plt.figure()
test_audio.plot_input()
plt.figure()
test_audio.plot_output()

test_audio.fit_linear()
test_audio.linear_model.summary()
linear_output = test_audio.transform_linear()
plt.figure()
plt.plot(linear_output)
test_audio.visualize_linear()

linear_weights = test_audio.linear_model.get_layer(name='dense_1').get_weights()

