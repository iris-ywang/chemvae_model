"""Module for supporting utility functions for the train_vae module."""

import tensorflow as tf
import subprocess


def print_gpu_usage():
    gpu_usage = subprocess.run(
        ['nvidia-smi', '--format=csv', '--query-gpu=utilization.gpu,memory.used'], stdout=subprocess.PIPE
    )
    print(gpu_usage.stdout.decode('utf-8'))


# Define a custom callback to print GPU usage
class GPUUsageCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_every=50):
        self.print_every = print_every

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.print_every == 0:  # Print every 'print_every' batches
            print(f"Batch {batch} GPU usage:")
            print_gpu_usage()

