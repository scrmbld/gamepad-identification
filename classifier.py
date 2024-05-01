# This module contains everything needed to run the classifier
# This includes the classifier itself as well as all of the
# functions needed to load & preprocess the data
#---------------------------------------------------------------

import os
import random

import pandas as pd
import numpy as np

# some default values
SPLIT=(0.6, 0.2, 0.2)
SEQ_LEN = 120

def get_filenames(data_path='dataset/velocities/'):
    """
    Return a list of all the filenames in the directory specified by data_path.
    """
    return os.listdir(data_path)
    

# load the input data
#---------------------------------------------------------------
def players_inputs(file_path='dataset/velocities/'):
    """
    Given a file path containing player input csv files, load them all in and sort them by player.
    Returns a dict in the form player_name: player_inputs where player_inputs is a pd.DataFrame.
    """
    all_files = get_filenames(file_path)
    print(all_files)

    games_inputs = {}
    for filename in all_files:
        fpath = file_path + filename
        inputs = pd.read_csv(fpath)
        games_inputs[filename.split('.')[0]] = inputs

    player_files = {}
    for filename in all_files:
        user = filename.split('_')[0]

        if user not in player_files.keys():
            player_files[user] = []
        
        player_files[user].append(filename.split('.')[0])
    
    inputs_by_player = {}

    for player in player_files.keys():
        # get the inputs from all the games played by a certain player
        player_games_inputs = [pd.read_csv(file_path + game + '.csv', index_col=None) for game in player_files[player]]
        player_inputs = pd.concat(player_games_inputs, ignore_index=True)
        
        inputs_by_player[player] = player_inputs
    
    return inputs_by_player

inputs_by_player = players_inputs('dataset/velocities/')

# tokenization of players
#---------------------------------------------------------------
players_set = list(inputs_by_player.keys())
players_set.sort() # ensure the mappings are the same every time

p_to_i = {key: i for i, key in enumerate(players_set)}
i_to_p = {i: key for key, i in p_to_i.items()}

def player_to_int(player):
    return p_to_i[player]

def int_to_player(i):
    return i_to_p[i]

def mapping():
    return p_to_i

# prepare the input data
#---------------------------------------------------------------
def seqs_from_df(df, player_id, seq_len=SEQ_LEN):
    """
    Arrange a dataframe of inputs into sequences of inputs of length seq_len.
    """
    seqs = []

    label = np.zeros(len(players_set))
    label[player_id] = 1.0

    for i in range(0, df.shape[0] - seq_len, int(seq_len*0.5)):
        seq_x = df.loc[i:i+seq_len-1]
        seqs.append((seq_x, label))
        
    return seqs


def split_ml_data(examples_list, split=SPLIT):
        """
        Split a list of input sequences into three sets based on the values of split.
        Returns three lists of input sequences.
        """
        split_indices = [0, 0, 0]
        for i in range(3):
            split_indices[i] = int(split[i] * len(examples_list))
            if i >= 1:
                split_indices[i] += split_indices[i-1]
        
        return (examples_list[0:split_indices[0]], examples_list[split_indices[0]:split_indices[1]], examples_list[split_indices[1]:split_indices[2]])

def split_set(ml_set):
        """
        Given a list of zipped example inputs and outputs, return two separate lists.
        """
        return (np.array([x.to_numpy(dtype=np.float32) for (x, _) in ml_set], dtype=np.float32), np.array([y for (_, y) in ml_set]))

def prepare_data(inputs_by_player, seq_len=SEQ_LEN, split=SPLIT):
    """
    Given a dict in which the keys are labels and the values are dataframes of inputs, return training, validation, and test data in which the dataframes have been organized into labelled sequences.
    """
    def label_data():
        labelled_ml_data = [seqs_from_df(inputs_by_player[p], player_to_int(p), seq_len) for p in players_set]
        labelled_ml_data = [seq for p in labelled_ml_data for seq in p] # flatten
        random.shuffle(labelled_ml_data)
        return labelled_ml_data

    labelled_ml_data = label_data()

    train, valid, test = split_ml_data(labelled_ml_data, split)

    train_x, train_y = split_set(train)
    valid_x, valid_y = split_set(valid)
    test_x, test_y = split_set(test)

    return ((train_x, train_y), (valid_x, valid_y), (test_x, test_y))

# the classifier
#---------------------------------------------------------------
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, ELU, Dropout, Dense, Concatenate, LSTM, GRU

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def createClassifier(width=3, seq_len=120):
    input_layer = Input(shape=(seq_len, width))
    conv1 = Conv1D(filters=32, kernel_size=7, strides=2, activation=ELU())(input_layer)
    conv2 = Conv1D(filters=32, kernel_size=3, strides=1, activation=ELU())(input_layer)

    catted = Concatenate(axis=1)([conv1, conv2])
    elu1 = ELU(32)(catted)
    conv3 = Conv1D(filters=32, kernel_size=2, strides=1, activation=ELU())(elu1)
    drop1 = Dropout(0.2)(conv3)

    gru1 = GRU(32, return_sequences=True)(drop1)
    gru2 = GRU(32)(gru1)
    drop2 = Dropout(0.2)(gru2)

    output = Dense(len(players_set), activation='softmax')(drop2)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

"""
def createClassifier(width=3, seq_len=180):
    input_layer = Input(shape=(seq_len, width))
    conv1 = Conv1D(filters=32, kernel_size=7, strides=2, activation=ELU())(input_layer)
    conv2 = Conv1D(filters=32, kernel_size=3, strides=1, activation=ELU())(input_layer)

    catted = Concatenate(axis=1)([conv1, conv2])
    elu1 = ELU(32)(catted)
    conv3 = Conv1D(filters=32, kernel_size=2, strides=1, activation=ELU())(elu1)
    conv4 = Conv1D(filters=32, kernel_size=2, strides=1, activation=ELU())(conv3)
    drop1 = Dropout(0.2)(conv4)

    gru1 = GRU(32, return_sequences=True)(drop1)
    gru2 = GRU(32)(gru1)
    drop2 = Dropout(0.2)(gru2)

    output = Dense(len(players_set), activation='softmax')(drop2)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
"""