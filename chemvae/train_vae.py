"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network

"""

# from gpu_utils import pick_gpu_lowest_memory
# gpu_free_number = str(pick_gpu_lowest_memory())
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_free_number)

import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
import yaml
import time
import os
import gc
import logging as lg

from keras import backend as K
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from chemvae import hyperparameters
from chemvae import mol_callbacks as mol_cb
from chemvae.models import encoder_model, load_encoder
from chemvae.models import decoder_model, load_decoder
from chemvae.models import property_predictor_model, load_property_predictor
from chemvae.models import variational_layers
from chemvae.train_utils import GPUUsageCallback, vectorize_data_chembl, generate_loop_batch_data_for_model_fit
from functools import partial
from keras.layers import Lambda
from chemvae.qsar import testing_encoder


#### Set up logging
# Get the TensorFlow logger
logging = lg.getLogger("tensorflow")

# Disable existing handlers for TensorFlow logger to prevent double output
logging.handlers.clear()
logging.propagate = False  # Prevent TensorFlow logs from going to the root logger

# Set up your custom handler for TensorFlow
ch = lg.StreamHandler()
ch.setLevel(lg.DEBUG)
formatter = lg.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logging.addHandler(ch)

# Set the logging level
logging.setLevel(lg.INFO)

# If necessary, set the root logger to a higher level to suppress additional output
lg.getLogger().setLevel(lg.WARNING)



# #### Set up GPU info
# logging.info(f"CUDA Version:  {tf.sysconfig.get_build_info()["cuda_version"]}")
# logging.info(f"CUDNN Version:  {tf.sysconfig.get_build_info()["cudnn_version"]}")
#
# # Check available GPUs
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     logging.info(f"GPUs available: {gpus} \n ")
# else:
#     logging.info("No GPU available! \n ")
#
# if gpus:
#     try:
#         # Set memory growth for each GPU (prevents TensorFlow from using all the GPU memory at once)
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#
#         # Make all GPUs visible to TensorFlow
#         tf.config.set_visible_devices(gpus, "GPU")
#
#         logging.info(f"Using GPUs: {gpus} \n")
#     except RuntimeError as e:
#         logging.info(f"Error setting GPUs: {e}")



def load_models(params):

    def identity(x):
        return tf.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params["kl_loss_weight"])

    if params["reload_model"] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # Decoder
    if params["do_tgru"]:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    if params["paired_output"]:
        # Method 1: original pair
        x_out = Lambda(identity, name="x_pred")(x_out)

        # Method 2: swapped pair element
        # Add the "swap_halves" operation on x_out to the model.
        # This is done to make the model output the same as the input,
        # but with the first half of the input swapped with the second half.
        # x_out = Lambda(swap_halves, name="x_pred")(x_out)

        # Method 3: interleave bits of the pair
        # x_out = Lambda(interleave_halves, name="x_pred")(x_out)

    else:
        x_out = Lambda(identity, name="x_pred")(x_out)

    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params["do_prop_pred"]:
        if params["reload_model"] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (("reg_prop_tasks" in params) and (len(params["reg_prop_tasks"]) > 0 ) and
                ("logit_prop_tasks" in params) and (len(params["logit_prop_tasks"]) > 0 )):

            reg_prop_pred, logit_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name="reg_prop_pred")(reg_prop_pred)
            logit_prop_pred = Lambda(identity, name="logit_prop_pred")(logit_prop_pred)
            model_outputs.extend([reg_prop_pred,  logit_prop_pred])

        # regression only scenario
        elif ("reg_prop_tasks" in params) and (len(params["reg_prop_tasks"]) > 0 ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name="reg_prop_pred")(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ("logit_prop_tasks" in params) and (len(params["logit_prop_tasks"]) > 0 ):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = Lambda(identity, name="logit_prop_pred")(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError("no logit tasks or regression tasks specified for property prediction")

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    logging.info(f"x_mean shape in kl_loss: {x_mean.get_shape()}")
    kl_loss = - 0.5 * \
        K.mean(1 + x_log_var - K.square(x_mean) -
              K.exp(x_log_var), axis=-1)
    return kl_loss


@tf.function
def compute_loss(AE_only_model, x, y):
    # AE_only_model has two outputs called x_pred and z_mean_log_var. model_losses tells the
    # model how to calculate the loss for each output.
    # Note z_mean_log_var is a concatenated tensor of z_mean and z_log_var.
    logits = AE_only_model(x, training=True)

    loss_kl = kl_loss(None, logits[1])
    loss_xent = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(x, logits[0]), axis=1)
    # Aggregate both losses
    return tf.reduce_mean(loss_kl + loss_xent)


def main_no_prop(params):

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    start_time = time.time()
    # Load data
    X_train_all, X_test_all = vectorize_data_chembl(params)

    # Set up multi-use of GPU
    n_gpu_to_use = params["n_gpu"]
    if params["model_fit_batch_size"] % n_gpu_to_use != 0:
        raise ValueError("Batch size must be divisible by the number of GPUs to use.")

    gpu_list = [f"GPU:{i}" for i in range(n_gpu_to_use)]
    strategy = tf.distribute.MirroredStrategy(gpu_list)
    logging.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():

        # if you don't have multiple GPUs, comment out the above line and uncomment and outdent the following line
        # if True:

        # Load model
        AE_only_model, encoder, decoder, kl_loss_var = load_models(params)

        # Compile models
        if params["optim"] == "adam":
            optim = Adam(lr=params["lr"], beta_1=params["momentum"], clipnorm=0.1)
        elif params["optim"] == "rmsprop":
            optim = RMSprop(lr=params["lr"], rho=params["momentum"])
        elif params["optim"] == "sgd":
            optim = SGD(lr=params["lr"], momentum=params["momentum"])
        else:
            raise NotImplemented("Please define valid optimizer")

        AE_only_model.compile(
            loss=lambda x, y: compute_loss(AE_only_model, x, y),
            optimizer=optim,
            metrics={"x_pred": ["categorical_accuracy", vae_anneal_metric]}
        )

        # Set up callbacks
        vae_sig_schedule = partial(
            mol_cb.sigmoid_schedule,
            slope=params["anneal_sigmod_slope"],
            start=params["vae_annealer_start"]
        )
        vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
                vae_sig_schedule, kl_loss_var, params["kl_loss_weight"], "vae" )

        csv_clb = CSVLogger(params["history_file"], append=False)
        gpu_usage_cb = GPUUsageCallback()
        callbacks = [vae_anneal_callback, csv_clb, gpu_usage_cb]

        # Split training loops over fit()
        batch_size_per_loop = params["loop_over_fit_batch_size"] if params["data_size_for_all_loops"] else len(X_train_all)
        n_batch_per_loop = int(params["data_size_for_all_loops"] // batch_size_per_loop) if params["data_size_for_all_loops"] else 1
        logging.info(f"Number of training batches: {n_batch_per_loop}")

        for batch_id in range(n_batch_per_loop):

            # Skip previously completed batches until the specified loop_over_fit_batch_id
            if params["loop_over_fit_batch_id"] is not None:
                batch_start_id = int(params["loop_over_fit_batch_id"])
                if batch_id < batch_start_id:
                    logging.info(f"\n Skipping Batch {batch_id} as the start "
                                 f"batch_id is specified at {batch_start_id} \n ")
                    continue

            logging.info(f"Start looped training batch: {batch_id}")


            ########################Training########################
            # Get batched data per loop
            X_train, X_test = generate_loop_batch_data_for_model_fit(
                if_paired=params["paired_output"],
                X_train_all=X_train_all,
                X_test_all=X_test_all,
                batch_id=batch_id,
                n_training_batch=n_batch_per_loop,
                batch_size=batch_size_per_loop,
            )

            keras_verbose = params["verbose_print"]

            AE_only_model.fit(
                x=X_train, y=X_train,
                batch_size=params["model_fit_batch_size"],
                epochs=params["epochs"],
                initial_epoch=params["prev_epochs"],
                callbacks=callbacks,
                verbose=keras_verbose,
                validation_data=[X_test, X_test]
            )

            logging.info(f"\n \n \n \n \n Note: Finished training batch {batch_id}. "
                         f"Current time: {datetime.today().strftime('%H_%M_%S__%d_%m_%Y')}."
                         f"Saving weights...")
            encoder.save(params["encoder_weights_file"])
            decoder.save(params["decoder_weights_file"])

            encoder.save(params["encoder_weights_file"][:-3] + f"_{(batch_id + 1) * batch_size_per_loop}.h5")
            decoder.save(params["decoder_weights_file"][:-3] + f"_{(batch_id + 1) * batch_size_per_loop}.h5")

            testing_encoder(params["encoder_weights_file"], X_test)
            del X_train
            del X_test
            gc.collect()

    logging.info(f"Time of run : {time.time() - start_time}")
    logging.info("**FINISHED**")
    return


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--exp_file",
    #                     help="experiment file", default="exp.json")
    # parser.add_argument("-d", "--directory",
    #                     help="exp directory", default=None)
    # args = vars(parser.parse_args())

    # args = {"exp_file": "../models/zinc/exp.json", "directory": None}
    args = {"exp_file": "../models/chembl_paired/exp.json", "directory": None}

    # args = {"exp_file": "/home/yw453/chemvae_model/models/zinc/exp.json", "directory": None}
    # args = {"exp_file": "/home/yw453/chemvae_model/models/chembl/exp.json", "directory": None}

    if args["directory"] is not None:
        args["exp_file"] = os.path.join(args["directory"], args["exp_file"])

    params = hyperparameters.load_params(args["exp_file"])
    logging.info(f"All params: {params}")

    main_no_prop(params)
