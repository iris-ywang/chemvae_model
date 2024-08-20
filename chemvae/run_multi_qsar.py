import numpy as np

from chemvae.qsar import main

if __name__ == '__main__':
    # for SA-VAE
    models_ids = [12600, 25200, 37800, 50400, 63000, 75600, 88200, 100800,
                  113400, 126000, 138600, 151200, 163800, 176400, 189000]
    # for PA-VAE
    # models_ids = [252, 504, 756, 1008, 1260, 1512, 1764, 2016, 2268, 2520,
    #               2772, 3024, 3276, 3528, 3780, 4032, 4284, 4536, 4788,
    #               5040, 5292, 5544, 5796, 6048, 6300, 6552, 6804]
    # create a list, models_ids, from 252 to 12348 at 252*4 interval (inclusive)
    # models_ids = list(range(252, 12348 + 252, 252*4))



    metrics_for_all_models = []
    for model_id in models_ids:
        encoder_weights_file = f"../models/zinc/zinc_encoder_iris2_{model_id}.h5"
        metrics_all_fold = main(model_train_size=model_id, encoder_file=encoder_weights_file)
        res = np.average(np.array(metrics_all_fold), axis=0)
        print("Average metrics: \n", res)

        metrics = [models_ids.index(model_id), model_id] + list(res)
        metrics_for_all_models.append(metrics)
        print(f"Model {model_id} done.")
    np.save(
        f"../models/zinc/averaged_metrics_for_all_zinc_encoder_iris2_testsize_500.npy",
        metrics_for_all_models
    )