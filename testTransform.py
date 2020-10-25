# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:00:25 2020

Script to test basic transform model

@author: freel
"""
import tensorflow as tf
import neuralTransform as nt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
from keras import backend as K
import gc

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#attempt to clear Keras models
K.clear_session()
gc.collect()

test_audio = nt.neuralTransform('C:/users/freel/Desktop/neuralImpulse/input_unprocessed/SL68Trial_Raw.aif',
                            'C:/users/freel/Desktop/neuralImpulse/input_processed/SL68Trial.aif',
                            96000,
                            .001,
                            1,
                            8192)

#start by graphing the input signal and output signal for reference
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('SL68 raw signal')
test_audio.plot_input()
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('SL68 processed through IR')
test_audio.plot_output()

#now fit a simple y=mx + b linear model where the input signal is multiplied
#by a constant and has a constant added. This is an absolute baseline toy model
print('fitting simple linear model')
test_audio.fit_linear()
test_audio.linear_model.summary()
linear_output = test_audio.transform_linear()
fig = plt.figure(dpi=300, figsize=(9,6))
librosa.display.waveplot(y=linear_output[:,0], sr=96000)
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
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_linear.wav', linear_output, 96000)
#fit a linear regression with scikit and compare

print('fitting linear multi-step model with 1000 sample lookback')
test_audio.fit_linear_multi(1000, 2048, 1)
test_audio.linear_multi_model.summary()
multi_step_weights = test_audio.linear_multi_model.get_layer(index=0).get_weights()
fig = plt.figure(dpi=300, figsize=(9,6))
sns.lineplot(data=multi_step_weights[0])

linear_multi_output = test_audio.transform_linear_multi()
fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('multi-step linear model output')
librosa.display.waveplot(y=linear_multi_output[:,0], sr=96000)

linear_multi_diff = np.transpose(test_audio.output_clip[0])[1000:] - linear_multi_output[:,0]
fig = plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('difference with linear multi-step model')
librosa.display.waveplot(y=linear_multi_diff, sr=96000)

#output to audio file
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_linear_multistep_1000.wav', linear_multi_output, 96000)

#test dense single-layer model
test_audio.fit_dense(64,2048,1)
test_audio.dense_model.summary()
dense_model_output = test_audio.transform_dense()

#how does single-layer dense model look?
fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('single-layer dense model output')
librosa.display.waveplot(y=dense_model_output[:,0], sr=96000)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_dense.wav', dense_model_output, 96000)

#test dense double layer model
test_audio.fit_dense2(128,32,2048,1)
test_audio.dense2_model.summary()
dense2_model_output = test_audio.transform_dense2()

#how does two-layer dense model look?
fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('double-layer dense model output')
librosa.display.waveplot(y=dense2_model_output[:,0], sr=96000)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_dense2.wav', dense2_model_output, 96000)

#test dense three-layer model
test_audio.fit_dense3(128,64,32,2048,1)
test_audio.dense3_model.summary()
dense3_model_output = test_audio.transform_dense3()

#how does three-layer dense model look?
fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('triple-layer dense model output')
librosa.display.waveplot(y=dense3_model_output[:,0], sr=96000)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_dense3.wav', dense3_model_output, 96000)

#test three layer model with rnn after first stage
test_audio.fit_rnn_dense3(64,32,16,8,2048,1)
test_audio.rnn_dense3_model.summary()
rnn_dense3_model_output = test_audio.transform_rnn_dense3()

fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('early rnn, triple-layer dense model output')
librosa.display.waveplot(y=rnn_dense3_model_output[:,0], sr=96000)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_rnn_dense3.wav', dense3_model_output, 96000)

#test three layer model with rnn as output
test_audio.fit_dense3_rnn(128,64,32,2048,1)
dense3_rnn_model_output = test_audio.transform_rnn_dense3()

fig=plt.figure(dpi=300, figsize=(9,6))
fig.suptitle('late rnn, triple-layer dense model output')
librosa.display.waveplot(y=rnn_dense3_model_output[:,0], sr=96000)
sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_dense3_rnn.wav', dense3_model_output, 96000)


# print('fitting linear multi-step model with 1000 sample lookback')
# test_audio.fit_linear_multi(1000, 1, 1)
# multi_step_weights = test_audio.linear_multi_model.get_layer(index=0).get_weights()
# fig = plt.figure(dpi=300, figsize=(9,6))
# sns.lineplot(data=multi_step_weights[0])

# linear_multi_output = test_audio.transform_linear_multi()
# sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_linear_multistep_1000.wav', linear_multi_output, 96000)

# test_audio.fit_linear_multi(6400)
# multi_step_weights = test_audio.linear_multi_model.get_layer(index=0).get_weights()
# fig = plt.figure(dpi=300, figsize=(9,6))
# sns.lineplot(data=multi_step_weights[0])

# linear_multi_output = test_audio.transform_linear_multi()
# sf.write('C:/users/freel/Desktop/neuralImpulse/output/SL68_linear_multistep_6400.wav', linear_multi_output, 96000)
