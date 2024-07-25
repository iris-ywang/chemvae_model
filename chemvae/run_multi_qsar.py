import numpy as np

from chemvae.qsar import main

if __name__ == '__main__':
    models_ids = [12600, 25200, 37800, 50400, 63000, 75600, 88200, 100800,
                  113400, 126000, 138600, 151200, 163800, 176400, 189000]
    metrics_for_all_models = []
    for model_id in models_ids:
        encoder_weights_file = f"../models/zinc/zinc_encoder_iris2_{model_id}.h5"
        metrics = main(model_train_size=model_id, encoder_file=encoder_weights_file)
        res = np.average(np.array(metrics), axis=0)
        print("Average metrics: \n", res)

        metrics = [models_ids.index(model_id), model_id] + list(res)
        metrics_for_all_models.append(metrics)
        print(f"Model {model_id} done.")
    np.save("../models/zinc/averaged_metrics_for_all_zinc_encoder_iris2.npy", metrics_for_all_models)