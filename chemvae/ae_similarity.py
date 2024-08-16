import logging

from chemvae.vae_utils import VAEUtils
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import jaccard_score

def vae_similarity(
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
    Xoh_r = vae_sa.decode(Z, standardise=False)
    X_r = vae_sa.one_hot_to_smiles(Xoh_r)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = mfpgen.GetFingerprint(X)
    fp_r = mfpgen.GetFingerprint(X_r)

    for i in range(size):

        print(f"Calculating similarity for molecule {i}...")
        print("Original smiles     :", X[i])
        print("Reconstructed smiles:", X_r[i])

        metrics_oh = jaccard_score(Xoh[i], Xoh_r[i], average='micro')
        metrics_fp = jaccard_score(fp[i], fp_r[i])
        print(f"Similarity (one-hot): {metrics_oh}")
        print(f"Similarity (fingerprint): {metrics_fp}")
        metrics.append([metrics_oh, metrics_fp])

    return metrics


def main():
    # user parameters
    size = 100
    model_train_size = 12600
    base_path = '../models/zinc/'
    test_idx_file_path = '../models/zinc/test_idx.npy'
    encoder_file = base_path + f'zinc_encoder_iris2_{model_train_size}.h5'
    decoder_file = base_path + f'zinc_decoder_iris2_{model_train_size}.h5'
    metrics_filename = f"sa_model_iris2_{model_train_size}_similarity_scores_testsize_{size}.npy"


    metrics = vae_similarity(
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
    main()
