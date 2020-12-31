#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:21:55 2020

@author: ryan
"""


import os
import numpy as np
import random
import tqdm
# Recurrent Neural Network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from scipy.io import wavfile
import pickle


FULL_SONG_SAMPLES_PREFIX = 'full_song_samples/'
TRAINING_ARRAYS_PREFIX = 'numpy_arrays_for_training/'
TARGET_SAMPLE_LENGTH = 50
SAMPLE_RATE = 44100
FEATURES = int(TARGET_SAMPLE_LENGTH /2)

input_files = os.listdir(FULL_SONG_SAMPLES_PREFIX)


def create_numpy_arrays_for_training(input_files):

    for file in input_files:
        samplerate, data = wavfile.read(f'{FULL_SONG_SAMPLES_PREFIX}{file}')
        file = file.split(sep = '.')
        file = file[0]
        with open(f'{TRAINING_ARRAYS_PREFIX}{file}.pkl','wb') as f:
            pickle.dump(data,f)


def open_and_load_training_array(fp):
    
    with open(fp,'rb') as f:
        data = pickle.load(f)
    
    return data

def create_random_sample(training_array):
    
    total_len = len(training_array)
    max_value = total_len - TARGET_SAMPLE_LENGTH - 1
    
    random_sample = random.randint(0,max_value)
    
    sample = np.array(training_array[random_sample: random_sample + \
                                     TARGET_SAMPLE_LENGTH])
    train_array = sample[0:int(TARGET_SAMPLE_LENGTH / 2)]
    target_array = sample[int(TARGET_SAMPLE_LENGTH / 2) : ]
    
    train_array = train_array.reshape(1,int(TARGET_SAMPLE_LENGTH / 2), 2)
    target_array = np.array(target_array)
    
    return train_array, target_array


def create_sample_set(training_array_files,batch_size = 1000):
    
    train_samples = []
    target_samples_left = []
    target_samples_right = []
    
    features = int(TARGET_SAMPLE_LENGTH / 2)
    
    for i in tqdm.tqdm(range(batch_size)):
            
        file = training_array_files[random.randint(0,len(training_array_files) - 1)]
            
            
        data = open_and_load_training_array(f'{TRAINING_ARRAYS_PREFIX}{file}')
        temp_train, temp_target = create_random_sample(data)
            
        train_samples.append(temp_train)
        target_samples_left.append(temp_target[:,0])
        target_samples_right.append(temp_target[:,1])
        
    train_samples = np.array(train_samples)
    train_samples = train_samples.reshape(batch_size, features,2)
        
    target_samples_right = np.array(target_samples_right)
    target_samples_left = np.array(target_samples_left)
        
        
    return train_samples, target_samples_right, target_samples_left

def batch_generator(training_array_files,batch_size,
                    steps):
    idx=1
    while True: 
        yield create_sample_set(training_array_files,batch_size)## Yields data
        if idx<steps:
            idx+=1
        else:
            idx=1

def create_model():

    print('setting up')
    inputs = Input(shape = (FEATURES, 2))
    layer_1 = LSTM(int(FEATURES *2),return_sequences = False, activation = 'relu')(inputs)
    layer_3 = Dense(int(FEATURES *2), activation = 'relu')(layer_1)
    layer_4 = Dense(int(FEATURES*2), activation = 'relu')(layer_3)
    left_output = Dense(FEATURES, name = 'left_output', activation = 'linear')(layer_4)
    right_output = Dense(FEATURES, name = 'right_output', activation = 'linear')(layer_4)
        
    
    print('creating model')
    model = Model(inputs = inputs, outputs = [left_output, right_output])
    print('compiling model')
    model.compile(optimizer = 'adam', loss = 'mse')
        
    return model

if __name__ == '__main__':
    
    input_files = os.listdir(FULL_SONG_SAMPLES_PREFIX)
    create_numpy_arrays_for_training(input_files)
    
    training_array_files = os.listdir(TRAINING_ARRAYS_PREFIX)
    
    model = create_model()
    
    for i in range(2):
        train_features, right_targets, left_targets = create_sample_set(training_array_files, batch_size = 10000)
        model.fit(train_features, [right_targets, left_targets], epochs = 10)
        
    pred_sample, left, right = create_sample_set(training_array_files, batch_size = 1)
    
    tester = model.predict(pred_sample)
    
    
    
    