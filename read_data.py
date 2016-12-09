#
# Routines for reading datasets
#
import pickle
import numpy as np
import glob
import gzip


def load_mnist(dirpath='./data/mnist'):
    """
    """
    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # Read the training and test set images and labels.
    X_train = load_mnist_images('%s/train-images-idx3-ubyte.gz' % dirpath)
    y_train = load_mnist_labels('%s/train-labels-idx1-ubyte.gz' % dirpath)
    X_test = load_mnist_images('%s/t10k-images-idx3-ubyte.gz' % dirpath)
    y_test = load_mnist_labels('%s/t10k-labels-idx1-ubyte.gz' % dirpath)

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_cifar10(dirpath='./data/cifar10'):
    """

    """
    # Load training data
    X, y = [], []
    for path in glob.glob('%s/data_batch_*' % dirpath):
        with open(path, 'rb') as f:
            batch = pickle.load(f)
        X.append(batch['data'])
        y.append(batch['labels'])
    X = np.concatenate(X) \
          .reshape(-1, 3, 32, 32) \
          .astype(np.float32)
    y = np.concatenate(y).astype(np.int32)

    # Split into training and validation sets
    np.random.seed(1234)
    ii = np.random.permutation(len(X))
    X_train = X[ii[1000:]]
    y_train = y[ii[1000:]]
    X_valid = X[ii[:1000]]
    y_valid = y[ii[:1000]]

    # Load test set
    path = '%s/test_batch' % dirpath
    with open(path, 'rb') as f:
        batch = pickle.load(f)
    X_test = batch['data'] \
             .reshape(-1, 3, 32, 32) \
             .astype(np.float32)
    y_test = np.array(batch['labels'], dtype=np.int32)

    # Normalize to zero mean and unity variance
    offset = np.mean(X_train, 0)
    scale = np.std(X_train, 0).clip(min=1)
    X_train = (X_train - offset) / scale
    X_valid = (X_valid - offset) / scale
    X_test = (X_test - offset) / scale

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_cifar100(dirpath='./data/cifar100'):
    raise NotImplementedError


def load_svhn(dirpath='./data/svhn'):
    raise NotImplementedError
