# from os import environ
# environ['KERAS_BACKEND'] = 'tensorflow'
# vae stuff
from chemvae.vae_utils import VAEUtils
import numpy as np
import pandas as pd

from chemvae import mol_utils as mu
# # import scientific py
# # rdkit stuff
# from rdkit.Chem import AllChem as Chem
# from rdkit.Chem import PandasTools
# # plotting stuff
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from IPython.display import SVG, display


########## For VAE-SA Experiments:

if __name__ == '__main__':
    # Load the VAE model
    vae_sa = VAEUtils(
        exp_file='../models/zinc/exp.json',
        if_load_decoder=False,
        test_idx_file='../models/zinc/test_idx_42.npy',
    )

    logp_task = "logP"
    qsar_size = 200

    Z = vae_sa.Z[:qsar_size]
    y = np.array(vae_sa.reg_tasks[logp_task])[:qsar_size]

    train_test = np.concatenate([y, Z], axis=1)


    print("haha")

