# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:00:25 2020

Script to test basic transform model

@author: freel
"""
import tensorflow as tf
import neuralOptimize as nt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf

from tensorflow.keras import backend as K
import gc

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#attempt to clear Keras models
K.clear_session()
gc.collect()

sample_rt = 48000
lookback = 2000

test_audio = nt.neuralTransform('C:/users/freel/Desktop/neuralImpulse/input_unprocessed/unprocessed_full_Nov1.wav',
                            'C:/users/freel/Desktop/neuralImpulse/input_processed/processed_full_Nov1.wav',
                            sample_rt,
                            learning_rt=.001,
                            epoch_count=20,
                            batch_sz=1024,
                            dropout_rate=.05)

#start by graphing the input signal and output signal for reference
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('SL68 raw signal')
test_audio.plot_input()
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('SL68 processed through IR')
test_audio.plot_output()

#now fit a simple y=mx + b linear model where the input signal is multiplied
#by a constant and has a constant added. This is an absolute baseline toy model
print('\nfitting simple linear model')
test_audio.fit_linear()
test_audio.linear_model.summary()
linear_output = test_audio.transform_linear()
fig = plt.figure(dpi=300, figsize=(9,6))
librosa.display.waveplot(y=linear_output[:,0], sr=sample_rt)
fig.suptitle('SL68 through pure one-step linear model')
test_audio.visualize_linear()

#what did we come up with?
linear_weights = test_audio.linear_model.get_layer(index=0).get_weights()
print('linear weights: ' + str(linear_weights[0]) + ', ' + str(linear_weights[1]))
#calculate differences between linear output and actual output
linear_diff = np.transpose(test_audio.output_clip[0]) - linear_output[:,0]
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('Model vs. actual difference for linear one-step model')
plt.plot(linear_diff)
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('Correlation between input and output in original signal')
sns.regplot(x=test_audio.input_clip[0][0:10000], y=test_audio.output_clip[0][0:10000])
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_linear.wav', linear_output, sample_rt)
#fit a linear regression with scikit and compare

print('\nfitting linear multi-step model with' + ' {}'.format(lookback) + ' sample lookback')
test_audio.fit_linear_multi(1000)
test_audio.linear_multi_model.summary()
multi_step_weights = test_audio.linear_multi_model.get_layer(index=0).get_weights()
fig = plt.figure(dpi=300, figsize=(9,6))
sns.lineplot(data=multi_step_weights[0])

linear_multi_output = test_audio.transform_linear_multi()
fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('multi-step linear model output')
librosa.display.waveplot(y=linear_multi_output[:,0], sr=sample_rt)

linear_multi_diff = np.transpose(test_audio.output_clip[0])[1000:] - linear_multi_output[:,0]
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('difference with linear multi-step model')
librosa.display.waveplot(y=linear_multi_diff, sr=sample_rt)

#output to audio file
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_linear_multistep_1000.wav', linear_multi_output, sample_rt)

print('\nfitting double-layer model with 1000 sample lookback')
#test dense double layer model
test_audio.fit_dense2(256,64)
test_audio.dense2_model.summary()
dense2_model_output = test_audio.transform_dense2()

#how does two-layer dense model look?
fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('double-layer dense model output')
librosa.display.waveplot(y=dense2_model_output[:,0], sr=sample_rt)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_dense2.wav', dense2_model_output, sample_rt)

#test three layer model with rnn after first stage
test_audio.fit_rnn_dense3(128,64,32,16)
test_audio.rnn_dense3_model.summary()
rnn_dense3_model_output = test_audio.transform_rnn_dense3()

fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('early rnn, triple-layer dense model output')
librosa.display.waveplot(y=rnn_dense3_model_output[:,0], sr=sample_rt)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_rnn_dense3.wav', rnn_dense3_model_output, sample_rt)

#test three layer model with lstm after first stage
test_audio.fit_lstm_dense3(128,64,32,16)
test_audio.lstm_dense3_model.summary()
lstm_dense3_model_output = test_audio.transform_lstm_dense3()

fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('early lstm, triple-layer dense model output')
librosa.display.waveplot(y=lstm_dense3_model_output[:,0], sr=sample_rt)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_lstm_dense3.wav', lstm_dense3_model_output, sample_rt)

#show the training history
test_audio.plot_errors()
plt.clear()