"""Module for supporting utility functions for the train_vae module."""


import pandas as pd
import numpy as np
import yaml

import logging as lg

from chemvae import mol_utils as mu

import tensorflow as tf
import subprocess


logging = lg.getLogger("tensorflow")


def print_gpu_usage():
    gpu_usage = subprocess.run(
        ['nvidia-smi', '--format=csv', '--query-gpu=utilization.gpu,memory.used'], stdout=subprocess.PIPE
    )
    print(gpu_usage.stdout.decode('utf-8'))


# Define a custom callback to print GPU usage
class GPUUsageCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_every=300):
        self.print_every = print_every

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.print_every == 0:  # Print every 'print_every' batches
            print(f"Batch {batch} GPU usage:")
            print_gpu_usage()


def vectorize_data_chembl(params, n_samples=None):

    # For Morgan FP, MAX_LEN = 1024.
    MAX_LEN = params['MAX_LEN']

    chembl_data = pd.read_csv(params['data_file'])
    logging.info(f'Training set size is {len(chembl_data)}')
    if params["paired_output"]:
        params['NCHARS'] = 2
    else:
        params['NCHARS'] = 1

    X = chembl_data.iloc[:, 2:].to_numpy(dtype=np.float32)

    # Shuffle the data
    np.random.seed(params['RAND_SEED'])
    rand_idx = np.arange(np.shape(X)[0])
    np.random.shuffle(rand_idx)

    # Set aside the validation set
    TRAIN_FRAC = 1 - params['val_split']
    num_train = int(X.shape[0] * TRAIN_FRAC)

    if num_train % params['batch_size'] != 0:
        num_train = num_train // params['batch_size'] * \
            params['batch_size']

    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train):]

    if 'test_idx_file' in params.keys():
        np.save(params['test_idx_file'], test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    logging.info(f'shape of input vector : {np.shape(X_train)}')
    logging.info('Training set size is {}, after filtering to max length of {}'.format(
        np.shape(X_train), MAX_LEN))

    return X_train, X_test


def vectorize_data(params, n_samples=None):
    # @out : Y_train /Y_test : each is list of datasets.
    #        i.e. if reg_tasks only : Y_train_reg = Y_train[0]
    #             if logit_tasks only : Y_train_logit = Y_train[0]
    #             if both reg and logit_tasks : Y_train_reg = Y_train[0], Y_train_reg = 1
    #             if no prop tasks : Y_train = []

    MAX_LEN = params['MAX_LEN']
    if params["paired_output"]:
        MAX_LEN = int(MAX_LEN / 2)

    CHARS = yaml.safe_load(open(params['char_file']))
    params['NCHARS'] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
    #INDICES_CHAR = dict((i, c) for i, c in enumerate(CHARS))

    ## Load data for properties
    if params['do_prop_pred'] and ('data_file' in params):
        if "data_normalization_out" in params:
            normalize_out = params["data_normalization_out"]
        else:
            normalize_out = None

        ################
        if ("reg_prop_tasks" in params) and ("logit_prop_tasks" in params):
            smiles, Y_reg, Y_logit = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], logit_tasks=params['logit_prop_tasks'],
                    normalize_out = normalize_out)
        elif "logit_prop_tasks" in params:
            smiles, Y_logit = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN,
                    logit_tasks=params['logit_prop_tasks'], normalize_out=normalize_out)
        elif "reg_prop_tasks" in params:
            smiles, Y_reg = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], normalize_out=normalize_out)
        else:
            raise ValueError("please sepcify logit and/or reg tasks")

    ## Load data if no properties
    else:
        smiles = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN)

    if 'limit_data' in params.keys():
        sample_idx = np.random.choice(np.arange(len(smiles)), params['limit_data'], replace=False)
        smiles=list(np.array(smiles)[sample_idx])
        if params['do_prop_pred'] and ('data_file' in params):
            if "reg_prop_tasks" in params:
                Y_reg =  Y_reg[sample_idx]
            if "logit_prop_tasks" in params:
                Y_logit =  Y_logit[sample_idx]

    logging.info(f'Training set size is {len(smiles)}')
    logging.info(f'first smiles: {smiles[0]}')
    logging.info(f'total chars: {NCHARS}')

    logging.info('Vectorization...')
    X = mu.smiles_to_hot(smiles, MAX_LEN, params[
                             'PADDING'], CHAR_INDICES, NCHARS)

    logging.info(f'Total Data size {X.shape[0]}')
    if np.shape(X)[0] % params['batch_size'] != 0:
        X = X[:np.shape(X)[0] // params['batch_size']
              * params['batch_size']]
        if params['do_prop_pred']:
            if "reg_prop_tasks" in params:
                Y_reg = Y_reg[:np.shape(Y_reg)[0] // params['batch_size']
                      * params['batch_size']]
            if "logit_prop_tasks" in params:
                Y_logit = Y_logit[:np.shape(Y_logit)[0] // params['batch_size']
                      * params['batch_size']]

    np.random.seed(params['RAND_SEED'])
    rand_idx = np.arange(np.shape(X)[0])
    np.random.shuffle(rand_idx)

    TRAIN_FRAC = 1 - params['val_split']
    num_train = int(X.shape[0] * TRAIN_FRAC)

    if num_train % params['batch_size'] != 0:
        num_train = num_train // params['batch_size'] * \
            params['batch_size']

    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train):]

    if 'test_idx_file' in params.keys():
        np.save(params['test_idx_file'], test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    logging.info(f'shape of input vector : {np.shape(X_train)}')
    logging.info('Training set size is {}, after filtering to max length of {}'.format(
        np.shape(X_train), MAX_LEN))

    if params['do_prop_pred']:
        # !# add Y_train and Y_test here
        Y_train = []
        Y_test = []
        if "reg_prop_tasks" in params:
            Y_reg_train, Y_reg_test = Y_reg[train_idx], Y_reg[test_idx]
            Y_train.append(Y_reg_train)
            Y_test.append(Y_reg_test)
        if "logit_prop_tasks" in params:
            Y_logit_train, Y_logit_test = Y_logit[train_idx], Y_logit[test_idx]
            Y_train.append(Y_logit_train)
            Y_test.append(Y_logit_test)

        return X_train, X_test, Y_train, Y_test

    else:
        return X_train, X_test


def interleave_halves(x: tf.Tensor):
    """Function to take in a KerasTensor of shape (dim_a, n) and
    return a tensor of that double the length of shape (2*dim_a, n),
    but with their elements intersected.

    For example, if the input keras tensor has its first half being
    [[1,2,3,4]] and the second half being [[10, 20, 30, 40]], the
    output will be [[1, 10, 2, 20, 3, 30, 4, 40]]."""
    dim_a = x.shape[1]
    half_dim_a = int(dim_a // 2)
    x_a = x[:, :half_dim_a]
    x_b = x[:, half_dim_a:]
    stacked = tf.stack([x_a, x_b], axis=2)

    interleaved_pair = tf.reshape(stacked, [tf.shape(x_a)[0], -1])
    return interleaved_pair


def swap_halves(x: tf.Tensor):
    """
    Function to take in a 3D (?) KerasTensor of shape (dim_a, dim_b) and
    return a tensor of the same shape, but the first half in dim_a
    swaps position with the second half by the 2nd (?) axis.
    """
    dim_a = x.shape[1]
    half_dim_a = dim_a // 2
    return tf.concat([x[half_dim_a:], x[:half_dim_a]], axis=0)


def make_pairs(train_set: np.array, test_set: np.array):
    if len(train_set.shape) == 2:
        train_set = np.expand_dims(train_set, axis=1)
        test_set = np.expand_dims(test_set, axis=1)
        extra_step = True

    train_set_size, train_set_length, train_set_dim= train_set.shape
    test_set_size, test_set_length, test_set_dim = test_set.shape

    train_set_pairs = np.zeros((
        train_set_size * train_set_size, 2*train_set_length, train_set_dim
    ))
    test_set_pairs = np.zeros((
        train_set_size * test_set_size * 2, 2*test_set_length, test_set_dim
    ))

    for i in range(train_set_size):
        for j in range(train_set_size):
            train_set_pairs[i * train_set_size + j] = np.concatenate((train_set[i], train_set[j]), axis=0)
    for i in range(test_set_size):
        for j in range(train_set_size):
            test_set_pairs[i * train_set_size * 2 + j] = np.concatenate((train_set[j], test_set[i]), axis=0)
            test_set_pairs[i * train_set_size * 2 + train_set_size + j] = np.concatenate((test_set[i], train_set[j]), axis=0)

    if extra_step == 1.0:
        train_set_pairs = train_set_pairs.swapaxes(1, 2)
        test_set_pairs = test_set_pairs.swapaxes(1, 2)

    return train_set_pairs, test_set_pairs


def generate_loop_batch_data_for_model_fit(
        if_paired,
        X_train_all, X_test_all,
        batch_id, n_training_batch, batch_size,
):
    """

    Args:
        if_paired: if paired_output is True, make pairs of the input data
        X_train_all: from loading all data * 0.9
        X_test_all:  from loading all data * 0.1
        batch_id: the nth loop / looped batch to go through
        n_training_batch: total number of loops required
        batch_size: the batch size per loop over model.fit()
        callbacks:

    Returns:
        X_train: training data for the loop
        X_test: test data for the loop
    """
    # Batch data per loop
    if batch_id == n_training_batch - 1:
        X_train = X_train_all[batch_id * batch_size:]
        X_test = X_test_all[batch_id * int(batch_size / 10):]
        logging.info(f"Training loop batch index is from {batch_id * batch_size} to {len(X_train_all)} \n"
                     f"Test loop batch index is from {batch_id * int(batch_size / 10)} to {len(X_test_all)}")
    else:
        X_train = X_train_all[batch_id * batch_size:(batch_id + 1) * batch_size]
        X_test = X_test_all[batch_id * int(batch_size / 10):(batch_id + 1) * int(batch_size / 10)]
        logging.info(f"Training loop batch index is from {batch_id * batch_size} to {(batch_id + 1) * batch_size} \n"
                     f"Test loop batch index is from {batch_id * int(batch_size / 10)} to {(batch_id + 1) * int(batch_size / 10)}")

    # if paired_output is True, make pairs of the input data
    if if_paired:
        X_train, X_test = make_pairs(X_train, X_test)
    logging.info(f"Size of training and test size: {X_train.shape}, {X_test.shape}")

    # checking for NaN values in the data
    if np.isnan(X_train).any():
        logging.warning("\n!!!!!!! \n NaN values in training data \n !!!!!!!\n")
    else:
        logging.info("No NaN values in training data")
    if np.isnan(X_test).any():
        logging.warning("\n!!!!!!!\n NaN values in test data\n !!!!!!!\n")
    else:
        logging.info("No NaN values in test data")

    return X_train, X_test
