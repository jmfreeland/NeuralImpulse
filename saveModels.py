# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:36:32 2020

@author: freel

test_audio.linear_model.save_weights('./checkpoints/linear_model')
test_audio.linear_multi_model.save_weights('./checkpoints/linear_multi_model')
test_audio.dense2_model.save_weights('./checkpoints/dense2_model')
test_audio.lstm_dense3_model.save_weights('./checkpoints/lstm_dense3_model')
test_audio.rnn_dense3_model.save_weights('./checkpoints/rnn_dense3_model')
"""

test_audio.linear_model.save('./deploy/linear_model', save_format='tf')
test_audio.linear_multi_model.save('./deploy/linear_multi_model', save_format='tf')
test_audio.dense2_model.save('./deploy/dense2_model', save_format='tf')
test_audio.lstm_dense3_model.save('./deploy/lstm_dense3_model', save_format='tf')
test_audio.rnn_dense3_model.save('./deploy/rnn_dense3_model', save_format='tf')

test_audio.linear_multi_model.save('./deploy/linear_model_multi.h5', include_optimizer=False)
