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
import numpy as np
import seaborn as sns

test_audio = nt.neuralTransform('C:/users/freel/Desktop/neuralImpulse/SL68Trial_Raw.aif',
                            'C:/users/freel/Desktop/neuralImpulse/SL68Trial.aif',
                            96000)

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

linear_weights = test_audio.linear_model.get_layer(index=0).get_weights()
linear_diff = np.transpose(test_audio.output_clip[0]) - linear_output[:,0]
plt.figure()
plt.plot(linear_diff)
sns.regplot(x=test_audio.input_clip[0][0:100000], y=test_audio.output_clip[0][0:100000])
#calculate differences between linear output and actual output
#fit a linear regression with scikit and compare

test_audio.fit_linear_multi(100)
multi_step_weights = test_audio.linear_multi_model.get_layer(index=0).get_weights()
sns.lineplot(data=multi_step_weights[0])

