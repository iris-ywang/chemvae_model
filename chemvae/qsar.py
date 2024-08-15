# from os import environ
# environ['KERAS_BACKEND'] = 'tensorflow'
# vae stuff
import logging

from chemvae.vae_utils import VAEUtils
import numpy as np
from functools import partial

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pairwise_formulation.pa_basics.import_data import kfold_splits
from pairwise_formulation.pairwise_data import PairwiseDataInfo
from pairwise_formulation.pairwise_model import build_ml_model, PairwiseRegModel
from keras.models import load_model

# # import scientific py
# # rdkit stuff
# from rdkit.Chem import AllChem as Chem
# from rdkit.Chem import PandasTools
# # plotting stuff
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from IPython.display import SVG, display


def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
    """
    Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
    estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]

    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairsc
    :param c2_test_pair_ids: list of tuples, each specifying samples IDs for a c2-type pair.
            * Y_pa_c2 and c2_test_pair_ids should match in position; their length should be the same.
    :param test_ids: list of int for test sample IDs
    :param y_true: np.array of true activity values of all samples
    :param Y_weighted: np.array of weighting of each Y_pred (for example, from model prediction probability)
    :return: np.array of estimated activity values for test set
    """
    if y_true is None:
        y_true = y_true
    if Y_weighted is None:  # linear arithmetic
        Y_weighted = np.ones((len(Y_pa_c2)))

    records = np.zeros((len(y_true)))
    weights = np.zeros((len(y_true)))

    for pair in range(len(Y_pa_c2)):
        ida, idb = c2_test_pair_ids[pair]
        delta_ab = Y_pa_c2[pair]
        weight = Y_weighted[pair]

        if ida in test_ids:
            # (test, train)
            weighted_estimate = (y_true[idb] + delta_ab) * weight
            records[ida] += weighted_estimate
            weights[ida] += weight

        elif idb in test_ids:
            # (train, test)
            weighted_estimate = (y_true[ida] - delta_ab) * weight
            records[idb] += weighted_estimate
            weights[idb] += weight

    return np.divide(records[test_ids], weights[test_ids])


def pairwise_differences_for_standard_approach(
        y_pred_all, pairwise_data_info
):
    Y_c2_abs_derived = []
    for pair_id in pairwise_data_info.c2_test_pair_ids:
        id_a, id_b = pair_id
        Y_c2_abs_derived.append(abs(y_pred_all[id_a] - y_pred_all[id_b]))

    Y_c3_abs_derived = []
    for pair_id in pairwise_data_info.c3_test_pair_ids:
        id_a, id_b = pair_id
        Y_c3_abs_derived.append(abs(y_pred_all[id_a] - y_pred_all[id_b]))

    return np.array(Y_c2_abs_derived), np.array(Y_c3_abs_derived)

# def metrics_evaluation(y_true, y_predict):
#     rho = spearmanr(y_true, y_predict, nan_policy="omit")[0]
#     mse = mean_squared_error(y_true, y_predict)
#     mae = mean_absolute_error(y_true, y_predict)
#     r2 = r2_score(y_true, y_predict)
#     return [rho, mse, mae, r2, np.nan, np.nan]


def testing_encoder(encoder_path, X_test):
    encoder = load_model(encoder_path)
    logging.info("Checking weights in the encoder model")
    if np.isnan(encoder.get_weights()[0]).any() or np.isnan(encoder.get_weights()[-1]).any():
        logging.warning("\n !!!-There are NaN values in the encoder model weights \n")
        error_in_weights = True
    else:
        error_in_weights = False
    Z_test_single = encoder.predict(X_test[0:1])[0]
    if np.isnan(Z_test_single).any():
        logging.warning("\n !!!-There are NaN values in the single encoder model prediction \n")
        error_in_prediction = True
    else:
        error_in_prediction = False
        print(Z_test_single)
    Z_test = encoder.predict(X_test[:50])[0]
    if np.isnan(Z_test).any():
        logging.warning("\n !!!-There are NaN values in the batch encoder model predictions \n")
        error_in_prediction = True
    else:
        error_in_prediction = False

    if error_in_weights or error_in_prediction:
        logging.warning("\n !!!!!!!!! "
                        "There are NaN values in the encoder model \n"
                        "!!!!!!!!!\n")


def vae_qsar_sa(qsar_size=200, logp_task="logP", encoder_file=None):
    vae_sa = VAEUtils(
        exp_file='../models/zinc/exp.json',
        if_load_decoder=False,
        test_idx_file='../models/zinc/test_idx.npy',
        encoder_file=encoder_file,
    )

    Z_sa = vae_sa.Z[-qsar_size:]  # the last [qsar_size] molecules of latent space representation for the test set
    y = np.array(vae_sa.reg_tasks[logp_task])[-qsar_size:]

    train_test = np.concatenate([y.reshape(len(y), 1), Z_sa], axis=1)
    ML_reg = RandomForestRegressor(random_state=2, n_jobs=10)
    train_test_splits_dict = kfold_splits(train_test=train_test, fold=10)

    metrics = []
    for fold_id, foldwise_data in train_test_splits_dict.items():
        train_set = foldwise_data['train_set']
        test_set = foldwise_data['test_set']
        pairwise_data = PairwiseDataInfo(
            train_set, test_set
        )
        # Run the standard approach
        _, y_sa_pred = build_ml_model(
            model=ML_reg,
            train_data=pairwise_data.train_ary,
            test_data=pairwise_data.test_ary
        )
        y_sa_pred_all = np.array(pairwise_data.y_true_all)
        y_sa_pred_all[pairwise_data.test_ids] = y_sa_pred


        # Evaluation
        Y_c2_true, Y_c3_true = pairwise_differences_for_standard_approach(
            y_pred_all=pairwise_data.y_true_all, pairwise_data_info=pairwise_data
        )
        Y_c2_sa, Y_c3_sa = pairwise_differences_for_standard_approach(
            y_pred_all=y_sa_pred_all, pairwise_data_info=pairwise_data
        )

        mse_c2_sa = mean_squared_error(Y_c2_true, Y_c2_sa)
        mse_c3_sa = mean_squared_error(Y_c3_true, Y_c3_sa)

        mae_c2_sa = mean_absolute_error(Y_c2_true, Y_c2_sa)
        mae_c3_sa = mean_absolute_error(Y_c3_true, Y_c3_sa)

        r2_c2_sa = r2_score(Y_c2_true, Y_c2_sa)
        r2_c3_sa = r2_score(Y_c3_true, Y_c3_sa)
        print(
            "\n Fold ID: ", fold_id,
            f"\n MSE C2 SA: {mse_c2_sa}",
            f"\n MSE C3 SA: {mse_c3_sa}",
            f"\n MAE C2 SA: {mae_c2_sa}",
            f"\n MAE C3 SA: {mae_c3_sa}",
            f"\n R2 C2 SA: {r2_c2_sa}",
            f"\n R2 C3 SA: {r2_c3_sa}"
        )
        metrics.append([mse_c2_sa, mse_c3_sa, mae_c2_sa, mae_c3_sa, r2_c2_sa, r2_c3_sa])

    return metrics


def get_encoder_pairwise_Z(
        pairwise_encoder, data, pair_ids, smile_length, n_chars,
        encoding_batch_size=5000,
):
    data = np.array(data)
    all_pairs = []

    n_batch = (len(pair_ids) // encoding_batch_size)
    if n_batch * encoding_batch_size < len(pair_ids):
        n_batch += 1

    one_hot_pairs = []
    delta_y_pairs = []
    n_pairs_compiled = 0
    for sample_id_a, sample_id_b in pair_ids:
        sample_a = np.reshape(data[sample_id_a, 1:], (smile_length, n_chars))
        sample_b = np.reshape(data[sample_id_b, 1:], (smile_length, n_chars))
        delta_y_ab = data[sample_id_a, 0] - data[sample_id_b, 0]
        pair_ab_one_hot = np.concatenate([sample_a, sample_b], axis=0)

        one_hot_pairs.append(pair_ab_one_hot)
        delta_y_pairs.append([delta_y_ab])
        n_pairs_compiled += 1

        if n_pairs_compiled == encoding_batch_size:
            Z_pa = pairwise_encoder(np.array(one_hot_pairs))
            y_Z_ab = np.concatenate((np.array(delta_y_pairs), Z_pa), axis=1)
            all_pairs += y_Z_ab.tolist()
            logging.info(f"Compiled {n_pairs_compiled} pairs")

            one_hot_pairs = []
            delta_y_pairs = []
            n_pairs_compiled = 0

    if n_pairs_compiled > 0:
        Z_pa = pairwise_encoder(np.array(one_hot_pairs))
        y_Z_ab = np.concatenate((np.array(delta_y_pairs), Z_pa), axis=1)
        all_pairs += y_Z_ab.tolist()

    return np.array(all_pairs)


def vae_qsar_pa(qsar_size=200, logp_task="logP", encoder_file=None):
    vae_pa = VAEUtils(
        exp_file='../models/zinc_paired_model/exp.json',
        if_load_decoder=False,
        test_idx_file='../models/zinc/test_idx.npy',
        encoder_file=encoder_file,
    )

    one_hot = vae_pa.Z[-qsar_size:]  # the last [qsar_size] molecules of ONE-HOT of SMILES for the test set
    y = np.array(vae_pa.reg_tasks[logp_task])[-qsar_size:]

    train_test = np.concatenate([y.reshape(len(y), 1), one_hot], axis=1)
    ML_reg = RandomForestRegressor(random_state=2, n_jobs=10)
    train_test_splits_dict = kfold_splits(train_test=train_test, fold=10)

    metrics = []
    for fold_id, foldwise_data in train_test_splits_dict.items():
        train_set = foldwise_data['train_set']
        test_set = foldwise_data['test_set']
        pairwise_data = PairwiseDataInfo(
        train_set, test_set
    )
        pa_encoder = partial(vae_pa.encode, standardize=False)
        pa_encoder_predictor = partial(
            get_encoder_pairwise_Z,
            pairwise_encoder=pa_encoder,
            smile_length=vae_pa.max_length,
            n_chars=vae_pa.params["NCHARS"],
        )

        # Run the pairwise approach
        pa_model = PairwiseRegModel(
            pairwise_data_info=pairwise_data,
            ML_reg=ML_reg,
            pairing_method=pa_encoder_predictor,
            search_model=None,
            test_batch_size=2000,
            train_batch_size=5000,
            pairing_params=None
        ).fit()
        Y_values = pa_model.predict(
            pairing_method=pa_encoder_predictor
        )

        # Evaluation
        Y_c2_true = Y_values.Y_pa_c2_nume_true
        Y_c3_true = Y_values.Y_pa_c3_nume_true
        Y_c2_pa = Y_values.Y_pa_c2_nume
        Y_c3_pa = Y_values.Y_pa_c3_nume

        mse_c2_pa = mean_squared_error(Y_c2_true, Y_c2_pa)
        mse_c3_pa = mean_squared_error(Y_c3_true, Y_c3_pa)

        mae_c2_pa = mean_absolute_error(Y_c2_true, Y_c2_pa)
        mae_c3_pa = mean_absolute_error(Y_c3_true, Y_c3_pa)

        r2_c2_pa = r2_score(Y_c2_true, Y_c2_pa)
        r2_c3_pa = r2_score(Y_c3_true, Y_c3_pa)

        print(
            "\n Fold ID: ", fold_id,
            f"\n MSE C2 SA: {mse_c2_pa}",
            f"\n MSE C3 SA: {mse_c3_pa}",
            f"\n MAE C2 SA: {mae_c2_pa}",
            f"\n MAE C3 SA: {mae_c3_pa}",
            f"\n R2 C2 SA: {r2_c2_pa}",
            f"\n R2 C3 SA: {r2_c3_pa}"
        )
        metrics.append([mse_c2_pa, mse_c3_pa, mae_c2_pa, mae_c3_pa, r2_c2_pa, r2_c3_pa])

    return metrics


def main(model_train_size=12600, encoder_file=None):
    logp_task = "logP"
    qsar_size = 500
    metrics_filename = f"pa_model_iris2_{str(model_train_size)}_testsize_{qsar_size}.npy"

    # metrics_sa = vae_qsar_sa(qsar_size=qsar_size, logp_task=logp_task, encoder_file=encoder_file)
    # np.save(f"../models/zinc/{metrics_filename}", metrics_sa)
    metrics_pa = vae_qsar_pa(qsar_size=qsar_size, logp_task=logp_task, encoder_file=encoder_file)
    np.save(f"../models/zinc_paired_model/{metrics_filename}", metrics_pa)

    # np.save([metrics_sa, metrics_pa], f"/qsar_outputs/{metrics_filename}")
    print("Finished!")
    return


if __name__ == '__main__':
    main(model_train_size='1512', encoder_file="../models/zinc_paired_model/zinc_paired_encoder2_1512.h5")
    # main(model_train_size='downloaded', encoder_file="../models/zinc/zinc_encoder.h5")