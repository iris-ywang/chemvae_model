"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network

"""

# from gpu_utils import pick_gpu_lowest_memory
# gpu_free_number = str(pick_gpu_lowest_memory())
#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)

import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
import yaml
import time
import os
import gc
import logging
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from chemvae import hyperparameters
from chemvae import mol_utils as mu
from chemvae import mol_callbacks as mol_cb
from keras.callbacks import CSVLogger
from chemvae.models import encoder_model, load_encoder
from chemvae.models import decoder_model, load_decoder
from chemvae.models import property_predictor_model, load_property_predictor
from chemvae.models import variational_layers
from functools import partial
from keras.layers import Lambda

# # Enable memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#
# # Set environment variable for memory allocator
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


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


def load_models(params):

    def identity(x):
        return K.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params['kl_loss_weight'])

    if params['reload_model'] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # Decoder
    if params['do_tgru']:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    if params['paired_output']:
        # Add the "swap_halves" operation on x_out to the model.
        # This is done to make the model output the same as the input,
        # but with the first half of the input swapped with the second half.
        x_out = Lambda(swap_halves, name='x_pred')(x_out)
    else:
        x_out = Lambda(identity, name='x_pred')(x_out)

    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params['do_prop_pred']:
        if params['reload_model'] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ) and
                ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 )):

            reg_prop_pred, logit_prop_pred   = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.extend([reg_prop_pred,  logit_prop_pred])

        # regression only scenario
        elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError('no logit tasks or regression tasks specified for property prediction')

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


# Function to take in a KerasTensor of shape (dim_a, dim_b) and
# return a tensor of the same shape, but the first half in dim_a
# swaps position with the second half. For example, if the input
# keras tensor is in shape (10,3), the output tensor will be of
# shape (10,3) but the first 5 rows will be swapped with the
# last 5 rows. So if you have
# [[1,2,3,4,5], [10,20,30,40,50], [6,7,8,9,10], [60,70,80,90,100]],
# the output will be
# [[6,7,8,9,10], [60,70,80,90,100], [1,2,3,4,5], [10,20,30,40,50]].
# Input tye: keras.engine.keras_tensor.KerasTensor
# Output type: keras.engine.keras_tensor.KerasTensor

def swap_halves(x: tf.Tensor):
    dim_a = x.shape[1]
    half_dim_a = dim_a // 2
    return tf.concat([x[half_dim_a:], x[:half_dim_a]], axis=0)


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    logging.info(f'x_mean shape in kl_loss: {x_mean.get_shape()}')
    kl_loss = - 0.5 * \
        K.mean(1 + x_log_var - K.square(x_mean) -
              K.exp(x_log_var), axis=-1)
    return kl_loss


def run_single_batch(
        AE_only_model, encoder, decoder,
        params,
        X_train_all, X_test_all,
        batch_id, n_training_batch, batch_size,
        callbacks
):
    # Batch data
    if batch_id == n_training_batch - 1:
        X_train = X_train_all[batch_id * batch_size:]
        X_test = X_test_all[batch_id * int(batch_size / 10):]
        logging.info(f"Training batch index is from {batch_id * batch_size} to {len(X_train_all)} \n"
                     f"Test batch index is from {batch_id * int(batch_size / 10)} to {len(X_test_all)}")
    else:
        X_train = X_train_all[batch_id * batch_size:(batch_id + 1) * batch_size]
        X_test = X_test_all[batch_id * int(batch_size / 10):(batch_id + 1) * int(batch_size / 10)]
        logging.info(f"Training batch index is from {batch_id * batch_size} to {(batch_id + 1) * batch_size} \n"
                     f"Test batch index is from {batch_id * int(batch_size / 10)} to {(batch_id + 1) * int(batch_size / 10)}")

    # if paired_output is True, make pairs of the input data
    if params["paired_output"]:
        X_train, X_test = make_pairs(X_train, X_test)
    logging.info(f"Size of training and test size: {X_train.shape}, {X_test.shape}")

    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}

    keras_verbose = params['verbose_print']

    AE_only_model.fit(
        x=X_train, y=model_train_targets,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        initial_epoch=params['prev_epochs'],
        callbacks=callbacks,
        verbose=keras_verbose,
        validation_data=[X_test, model_test_targets]
    )

    logging.info(f"\n \n \n \n \n Note: Finished training batch {batch_id}. "
                 f"Current time: {datetime.today().strftime('%H_%M_%S__%d_%m_%Y')}."
                 f"Saving weights...")
    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    del X_train
    del X_test
    gc.collect()
    return AE_only_model


def main_no_prop(params):

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var


    start_time = time.time()

    X_train_all, X_test_all = vectorize_data(params)
    AE_only_model, encoder, decoder, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}
    xent_loss_weight = K.variable(params['xent_loss_weight'])
    AE_only_model.compile(
        loss=model_losses,
        loss_weights=[xent_loss_weight, kl_loss_var],
        optimizer=optim,
        metrics={'x_pred': ['categorical_accuracy', vae_anneal_metric]}
    )

    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)
    callbacks = [ vae_anneal_callback, csv_clb]

    # Load data
    if params["data_size"] is None:
        batch_size = len(X_train_all)
        n_training_batch = 1
    else:
        batch_size = params["training_batch_size"]
        n_training_batch = int(params["data_size"] // batch_size)
    logging.info(f'Number of training batches: {n_training_batch}')

    for batch_id in range(n_training_batch):
        if params["batch_id"] is not None:
            batch_start_id = int(params["batch_id"])
            if batch_id < batch_start_id:
                logging.info(f'\n Skipping Batch {batch_id} as the start '
                             f'batch_id is specified at {batch_start_id} \n ')
                continue
        logging.info(f'Training batch: {batch_id}')
        AE_only_model = run_single_batch(
            AE_only_model, encoder, decoder,
            params,
            X_train_all, X_test_all,
            batch_id, n_training_batch, batch_size,
            callbacks
        )

    logging.info(f'Time of run : {time.time() - start_time}')
    logging.info('**FINISHED**')
    return


def main_property_run(params):
    start_time = time.time()

    # load data
    X_train, X_test, Y_train, Y_test = vectorize_data(params)

    # load full models:
    AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    ae_loss_weight = 1. - params['prop_pred_loss_weight']
    model_loss_weights = {
                    'x_pred': ae_loss_weight*xent_loss_weight,
                    'z_mean_log_var':   ae_loss_weight*kl_loss_var}

    prop_pred_loss_weight = params['prop_pred_loss_weight']


    if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
        model_train_targets['reg_prop_pred'] = Y_train[0]
        model_test_targets['reg_prop_pred'] = Y_test[0]
        model_losses['reg_prop_pred'] = params['reg_prop_pred_loss']
        model_loss_weights['reg_prop_pred'] = prop_pred_loss_weight
    if ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
        if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            model_train_targets['logit_prop_pred'] = Y_train[1]
            model_test_targets['logit_prop_pred'] = Y_test[1]
        else:
            model_train_targets['logit_prop_pred'] = Y_train[0]
            model_test_targets['logit_prop_pred'] = Y_test[0]
        model_losses['logit_prop_pred'] = params['logit_prop_pred_loss']
        model_loss_weights['logit_prop_pred'] = prop_pred_loss_weight


    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)

    callbacks = [ vae_anneal_callback, csv_clb]
    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    # control verbose output
    keras_verbose = params['verbose_print']

    if 'checkpoint_path' in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                params=params, prop_pred_model = property_predictor,save_best_only=False))

    AE_PP_model.compile(loss=model_losses,
               loss_weights=model_loss_weights,
               optimizer=optim,
               metrics={'x_pred': ['categorical_accuracy',
                    vae_anneal_metric]})


    AE_PP_model.fit(X_train, model_train_targets,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         initial_epoch=params['prev_epochs'],
                         callbacks=callbacks,
                         verbose=keras_verbose,
         validation_data=[X_test, model_test_targets]
     )

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    property_predictor.save(params['prop_pred_weights_file'])

    logging.info(f'time of run : {time.time() - start_time}')
    logging.info('**FINISHED**')

    return


# Function to take two inputs, a train set and a test set of np.arrays of 3 dimensions, and return the pairwise version of the train set and test set.
# For the pairwise train set, the function will find all the permutation pairs of the input train set, and each pair will be represented by the concatenation of the input sample i and input sample j.
# For example, if the input train set is of shape (2, 1, 3), the output train set will be of shape (2*2, 2, 3).
# Say the input training sample is [[[1,2,3]], [[4,5,6]]].
# The pairs in the output training set will be [[1,2,3], [4,5,6]], [[1,2,3], [1,2,3]], [[4,5,6], [1,2,3]], [[4,5,6], [4,5,6]].
# For the pairwise test set, the function will find the combinatorial pairs of the input test set, and each pair will be represented by the concatenation of the input train sample i and input test sample j.
# For example, if the input train set is of shape (2, 1, 3) and the input test set is of shape (3, 1, 3), the output test set will be of shape (3*2*2, 2, 3).
# Say the input train set is [[[1,2,3]], [[4,5,6]]], and the input test set is [[[7,8,9]], [[10,11,12]],[[13,14,15]]].
# The pairs in the output test set will be [[1,2,3], [7,8,9]], [[1,2,3], [10,11,12]], [[1,2,3], [13,14,15]], [[4,5,6], [7,8,9]], [[4,5,6], [10,11,12]], [[4,5,6], [13,14,15]], and their reverse pairs ([[7,8,9],[1,2,3]], [[10,11,12],[1,2,3]], [[13,14,15],[1,2,3]], [[7,8,9],[4,5,6]], [[10,11,12],[4,5,6]], [[13,14,15],[4,5,6]]).
# Input type: np.array
# Output type: np.array

def make_pairs(train_set: np.array, test_set: np.array):
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

    return train_set_pairs, test_set_pairs




if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--exp_file',
    #                     help='experiment file', default='exp.json')
    # parser.add_argument('-d', '--directory',
    #                     help='exp directory', default=None)
    # args = vars(parser.parse_args())
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    args = {'exp_file': '../models/zinc_paired_model/exp.json', 'directory': None}
    # args = {'exp_file': '../models/zinc/exp.json', 'directory': None}

    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    params = hyperparameters.load_params(args['exp_file'])
    logging.info(f"All params: {params}")

    if params['do_prop_pred'] :
        main_property_run(params)
    else:
        main_no_prop(params)
