# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:34:37 2020

@author: freel
"""

import gc
gc.collect()

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