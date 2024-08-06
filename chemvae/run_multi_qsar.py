import numpy as np

from chemvae.qsar import main

if __name__ == '__main__':
    # for SA-VAE
    # models_ids = [12600, 25200, 37800, 50400, 63000, 75600, 88200, 100800,
    #               113400, 126000, 138600, 151200, 163800, 176400, 189000]
    # for PA-VAE
    models_ids = [252, 504, 756, 1008, 1260, 1512, 1764, 2016, 2268, 2520,
                  2772, 3024, 3276, 3528, 3780, 4032, 4284, 4536, 4788,
                  5040, 5292, 5544, 5796, 6048, 6300, 6552, 6804]

    metrics_for_all_models = []
    for model_id in models_ids:
        encoder_weights_file = f"../models/zinc_paired_model/zinc_paired_encoder_iris2_{model_id}.h5"
        metrics = main(model_train_size=model_id, encoder_file=encoder_weights_file)
        res = np.average(np.array(metrics), axis=0)
        print("Average metrics: \n", res)

        metrics = [models_ids.index(model_id), model_id] + list(res)
        metrics_for_all_models.append(metrics)
        print(f"Model {model_id} done.")
    np.save(
        f"../models/zinc_paired_model/averaged_metrics_for_zinc_paired_encoder_iris2_{model_id[0]}_to_{model_id[-1]}.npy",
        metrics_for_all_models
    )