import logging

from chemvae.vae_utils import VAEUtils
import numpy as np
from functools import partial

from sklearn.metrics import jaccard_score

def vae_sa_similarity(
        size=200,
        encoder_file=None,
        decoder_file=None,
        test_idx_file='../models/zinc/test_idx.npy',
        exp_file='../models/zinc/exp.json'
    ):
    vae_sa = VAEUtils(
        exp_file=exp_file,
        if_load_decoder=True,
        test_idx_file=test_idx_file,
        encoder_file=encoder_file,
        decoder_file=decoder_file,
    )

    metrics = []
    X = vae_sa.smiles_for_encoding[-size:]
    Xoh = vae_sa.smiles_one_hot_for_encoding[-size:]
    Z = vae_sa.Z[-size:]
    Xoh_r = vae_sa.decode(Z, standardized=False)
    X_r = vae_sa.hot_to_smiles(Xoh_r)

    for i in range(size):

        print(f"Calculating similarity for molecule {i}...")
        print("Original smiles     :", X[i])
        print("Reconstructed smiles:", X_r[i])

        metrics_oh = jaccard_score(np.round(Xoh[i]), np.round(Xoh_r[i]), average='micro')

        print(f"Similarity (one-hot): {metrics_oh}")
        metrics.append(metrics_oh)

    return metrics



def vae_pa_similarity(
        size=200,
        encoder_file=None,
        decoder_file=None,
        test_idx_file='../models/zinc/test_idx.npy',
        exp_file='../models/zinc_paired_model/exp.json'
    ):

    vae_pa = VAEUtils(
        exp_file=exp_file,
        if_load_decoder=True,
        test_idx_file=test_idx_file,
        encoder_file=encoder_file,
        decoder_file=decoder_file,
    )

    metrics = []
    X = vae_pa.smiles_for_encoding[-size:]
    Xoh = vae_pa.smiles_one_hot_for_encoding[-size:]
    # Z = vae_pa.Z[-size:]  # Z is just flattened one-hot

    metrics_dict = {s: [] for s in range(size)}

    for a in range(size):
        print(f"\n Running for molecule {a} of original smile: {X[a]}")
        sample_a = Xoh[a]

        oh_pairs_for_sample_a = []
        for b in range(size):
            sample_x = Xoh[b]
            pair_ab_one_hot = np.concatenate([sample_a, sample_x], axis=0)
            oh_pairs_for_sample_a.append(pair_ab_one_hot)

        Z_pa_sample_a = vae_pa.encode(np.array(oh_pairs_for_sample_a), standardize=False)
        Xoh_r_pa = vae_pa.decode(Z_pa_sample_a, standardized=False)  # (qsar_size, 240, 35)


        Xoh_r_as = Xoh_r_pa[:, :120, :]  # repeated Xoh for sample a
        Xoh_r_xs = Xoh_r_pa[:, 120:, :]  # One decoded Xoh for each sample

        print(f"Calculating average similarity for molecule {a}.")
        for i in range(size):
            Xoh_r_a = Xoh_r_as[i]
            metrics_oh_a = jaccard_score(np.round(Xoh[a]), np.round(Xoh_r_a), average='micro')
            metrics_dict[a].append(metrics_oh_a)

            Xoh_r_x = Xoh_r_xs[i]
            metrics_oh_x = jaccard_score(np.round(Xoh[i]), np.round(Xoh_r_x), average='micro')
            metrics_dict[i].append(metrics_oh_x)


            if i % 10 == 0:
                print(f"For molecule {a} and its repeated decoded subsample...")
                print("Original smiles     :", X[a])
                print("Reconstructed smiles:", vae_pa.hot_to_smiles(np.round(Xoh_r_a)))
                print(f"Calculating similarity score for molecule {a} "
                      f"and {i}th decoded molecule {a} from pairs: {metrics_oh_a} \n")

                print(f"For molecule {i} and its occurence as decoded subsample...")
                print("Original smiles     :", X[i])
                print("Reconstructed smiles:", vae_pa.hot_to_smiles(np.round(Xoh_r_x)))
                print(f"Calculating similarity score for molecule {i} "
                      f"and decoded molecule {i} from pairs: {metrics_oh_x}")


    for l in metrics_dict.values():
        metrics.append(np.mean(l))

    return metrics


def main_sa():
    # user parameters
    size = 100
    model_train_size = 12600
    base_path = '../models/zinc/'
    test_idx_file_path = '../models/zinc/test_idx.npy'
    encoder_file = base_path + f'zinc_encoder_iris2_{model_train_size}.h5'
    decoder_file = base_path + f'zinc_decoder_iris2_{model_train_size}.h5'
    metrics_filename = f"sa_model_iris2_{model_train_size}_similarity_scores_testsize_{size}.npy"


    metrics = vae_sa_similarity(
        size=size,
        encoder_file=encoder_file,
        decoder_file=decoder_file,
        test_idx_file=test_idx_file_path,
        exp_file=base_path + 'exp.json',
    )
    np.save(base_path + f"{metrics_filename}", metrics)
    print("Finished!")

    return metrics



def main_pa():
    # user parameters
    size = 100
    model_train_size = 1512
    base_path = '../models/zinc_paired_model/'
    test_idx_file_path = '../models/zinc/test_idx.npy'
    encoder_file = base_path + f'zinc_paired_encoder2_{model_train_size}.h5'
    decoder_file = base_path + f'zinc_paired_decoder2_{model_train_size}.h5'
    metrics_filename = f"pa_model_iris2_{model_train_size}_similarity_scores_testsize_{size}.npy"


    metrics = vae_pa_similarity(
        size=size,
        encoder_file=encoder_file,
        decoder_file=decoder_file,
        test_idx_file=test_idx_file_path,
        exp_file=base_path + 'exp.json',
    )
    np.save(base_path + f"{metrics_filename}", metrics)
    print("Finished!")

    return metrics



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    # main_sa()
    main_pa()
