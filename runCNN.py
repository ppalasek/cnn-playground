from __future__ import print_function
import sys
import os
import time
import argparse
import numpy as np
import theano
import theano.tensor as T
import lasagne
from aux import iterate_minibatches, create_trained_models_dir
from read_data import load_cifar10
from cnn_models import build_ccfff_model


def main():
    """
    ################################################################################
    runCNN: A framework for training a Convolutional Neural Network (CNN) on one of
    the following datasets:
        - MNIST
        - CIFAR-10
        - CIFAR-100
        - SVHN
    based on one of the following architectures:
        - 'ccfff' : A simple architecture consisting of 2 convolution and 3 fully-
                    connected layers (see build_ccfff_model() in cnn_models.py)
        - To be added more architectures (like ResNet, WRN, VGG)


    Run `runCNN.py -h` for more details on command line arguments.

    Example: If you want to




    ################################################################################
    """
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-d", "--dataset", type=str, choices=['mnist', 'cifar10', 'cifar100', 'svhn'], default='cifar10', help="choose dataset")
    parser.add_argument("-a", "--architecture", type=str, choices=['ccfff'], default="ccfff", help="choose CNN architecture")
    parser.add_argument("-e", "--num_epochs", type=int, default=10, help="set number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="set batch size")
    parser.add_argument("-l", '--loss', type=str, choices=['cross_entropy', 'hinge'], default="cross_entropy", help="choose loss (objective) function")
    parser.add_argument("-s", "--save_model", action="store_true", help="save model (weights)")
    args = parser.parse_args()

    # Brief description of ...
    if args.verbose:
        print(" # Training a CNN on %s with: " % args.dataset)
        print("    -- Architecture     : %s" % args.architecture)
        print("    -- Number of epochs : %d" % args.num_epochs)
        print("    -- Batch size       : %d" % args.batch_size)
        print("    -- Loss function    : %s" % args.loss)

    # Load the dataset
    print(" # Loading data...", end="")
    sys.stdout.flush()
    if args.dataset == "mnist":
        raise NotImplementedError
    if args.dataset == "cifar10":
        # Load dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
        data_shape = (None, 3, 32, 32)
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
    if args.dataset == "cifar100":
        raise NotImplementedError
    if args.dataset == "svhn":
        raise NotImplementedError
    print("Done!")

    # Build the CNN model
    print(" # Building cnn model and compiling functions...", end="")
    sys.stdout.flush()
    if args.architecture == "ccfff":
        network = build_ccfff_model(input_var=input_var, data_shape=data_shape)
    if args.architecture == "":
        raise NotImplementedError
    print("Done!")

    if args.verbose:
        print(" # Number of parameters in model: %d"
              % lasagne.layers.count_params(network, trainable=True))

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    if args.loss == "cross_entropy":
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    if args.loss == "hinge":
        loss = lasagne.objectives.multiclass_hinge_loss(prediction, target_var)
    loss = loss.mean()

    # Add weight decay
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss += l2_penalty

    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    if args.loss == "cross_entropy":
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    if args.loss == "hinge":
        test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Start training
    print(" # Starting training...", end="")
    sys.stdout.flush()
    # We iterate over epochs:
    for epoch in range(args.num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, args.batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        # Adjust learning rate (after 41 and 61 epochs)
        if (epoch + 1) == 41 or (epoch + 1) == 61:
            new_lr = sh_lr.get_value() * 0.1
            if args.verbose:
                print("New LR:" + str(new_lr))
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

    # Save trained model
    if args.save_model:
        # Create trained model dir
        trained_models_dir = "./trained_models"
        if not os.path.exists(trained_models_dir):
            os.makedirs(trained_models_dir)
        # Trained model filename format:
        # <dataset>_<architecture>_<num_epochs>_<batch_size>_<loss>_model.npz
        model_filename = "%s/%s_%s_%d_%d_%s_model.npz" % \
                         (trained_models_dir, args.dataset,
                          args.architecture, args.num_epochs,
                          args.batch_size, args.loss)
        np.savez(model_filename, *lasagne.layers.get_all_param_values(network))

    # Calculate validation error of model:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, args.batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))


if __name__ == "__main__":
    main()