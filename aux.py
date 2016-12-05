import numpy as np
import lasagne
import os


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Batch iterator
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def create_trained_models_dir(network, model_filename):
    """

    """
    trained_models_dir = "./trained_models"
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)

    np.savez(model_filename, *lasagne.layers.get_all_param_values(network))
