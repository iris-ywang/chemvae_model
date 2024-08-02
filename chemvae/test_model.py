import numpy as np
from chemvae.qsar import testing_encoder

def quick_test_encoder(encoder_path):
    X_test = np.random.randint(2, size=(100, 240, 35))
    testing_encoder(encoder_path, X_test)

if __name__ == '__main__':
    quick_test_encoder()