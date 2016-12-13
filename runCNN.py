from __future__ import print_function
import sys
import os
import time
import argparse
import numpy as np
import theano
import theano.tensor as T
import lasagne
from aux import iterate_minibatches
from read_data import load_mnist, load_cifar10, load_cifar100, load_svhn
from cnn_models import build_ccfff_model, build_ccffsvm_model, build_vgg5, build_vgg5_svm


def main():
    """
    runCNN: A framework for training a Convolutional Neural Network (CNN) on one of
    the following datasets:
        - MNIST
        - CIFAR-10
        - CIFAR-100
        - SVHN
    based on one of the following architectures:
        - 'ccfff'   : A simple architecture consisting of 2 convolution and 3 fully-
                      connected layers (see `build_ccfff_model()` in `cnn_models.py`)
        - 'ccffsvm' : Similar to 'ccfff' ...
        - To be added more architectures (like ResNet, WRN, VGG)

    Run `runCNN.py -h` for more details on command line arguments.
    ---------
    Example:

        python runCNN.py -d 'cifar10' -a 'ccfff' -e 100 -b 256 -l 'hinge' -s

        Train/evaluate a CNN with the 'ccfff' architecture on CIFAR-10. The number of
        epochs and the batch size are equal to 100 and 256, respectively, while the
        multi-class hinge loss will be used as an objective function. The trained
        model will be saved at
                `./trained_models/cifar10_ccfff_100_256_hinge_model.npz`.

    """
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-d", "--dataset", type=str, choices=['mnist', 'cifar10', 'cifar100', 'svhn'], default='cifar10', help="choose dataset")
    parser.add_argument("-e", "--num_epochs", type=int, default=100, help="set number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="set batch size")
    parser.add_argument("-m", "--save_model", action="store_true", help="save model (./trained_models/)")
    parser.add_argument("-r", "--save_results", action="store_true", help="save results (./results/)")
    parser.add_argument("-i", "--iter", type=int, default=1, help="iteration")
    parser.add_argument("-a", "--architecture", type=str,
                        choices=['ccfff-ap',
                                 'ccfff-mp',
                                 'ccffsvm-ap',
                                 'ccffsvm-mp',
                                 'vgg5',
                                 'vgg5-bn',
                                 'vgg16'],
                        default="ccffsvm-mp",
                        help="choose CNN architecture")
    args = parser.parse_args()

    # Print a brief description of the experiment
    print(" # Training a CNN on %s with: " % args.dataset)
    print("    -- Architecture     : %s" % args.architecture)
    print("    -- Number of epochs : %d" % args.num_epochs)
    print("    -- Batch size       : %d" % args.batch_size)
    print("    -- Iteration        : %d" % args.iter)

    # Load the dataset
    print(" # Loading data...", end="")
    sys.stdout.flush()
    if args.dataset == "mnist":
        # Load MNIST dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
        data_shape = (None, 1, 28, 28)
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

    elif args.dataset == "cifar10":
        # Load CIFAR-10 dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
        data_shape = (None, 3, 32, 32)
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

    elif args.dataset == "cifar100":
        # TODO: Load CIFAR-100 dataset
        # TODO: Prepare Theano variables for inputs and targets
        raise NotImplementedError

    elif args.dataset == "svhn":
        # TODO: SVHN dataset
        # TODO: Prepare Theano variables for inputs and targets
        raise NotImplementedError
    print("Done!")

    # Build the CNN model
    print(" # Building network model and compiling functions...", end="")
    sys.stdout.flush()

    # --------------------
    # Architectures: CCFFF
    # --------------------
    if args.architecture == 'ccfff-ap':
        network = build_ccfff_model(input_var=input_var, data_shape=data_shape, pool_mode='average_inc_pad')

    elif args.architecture == 'ccfff-mp':
        network = build_ccfff_model(input_var=input_var, data_shape=data_shape, pool_mode='max')

    # ----------------------
    # Architectures: CCFFSVM
    # ----------------------
    elif args.architecture == 'ccffsvm-ap':
        network = build_ccffsvm_model(input_var=input_var, data_shape=data_shape, pool_mode='average_inc_pad')

    elif args.architecture == 'ccffsvm-mp':
        network = build_ccffsvm_model(input_var=input_var, data_shape=data_shape, pool_mode='max')

    # -----------------------
    # Architectures: VGG-like
    # -----------------------
    elif args.architecture == "vgg5":
        network = build_vgg5(input_var=input_var, data_shape=data_shape, do_batch_norm=False)

    elif args.architecture == "vgg5-bn":
        network = build_vgg5(input_var=input_var, data_shape=data_shape, do_batch_norm=True)

    elif args.architecture == "vgg5-svm":
        network = build_vgg5_svm(input_var=input_var, data_shape=data_shape, do_batch_norm=False)

    elif args.architecture == "vgg5-bn-svm":
        network = build_vgg5_svm(input_var=input_var, data_shape=data_shape, do_batch_norm=True)

    # -----------------------------------
    # Architectures: To be added more ...
    # -----------------------------------
    elif args.architecture == "":
        raise NotImplementedError

    #
    # So Far, So Good
    #

    # Create a loss expression for training
    if "svm" in args.architecture:
        scores = lasagne.layers.get_output(network)
        loss = network.get_one_vs_all_cost_from_scores(scores, target_var)
    else:
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Add weight decay
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss += l2_penalty

    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    LRs = {'ccfff-ap': {1: 0.1},
           'ccfff-mp': {1: 0.1},
           'ccffsvm-ap': {1: 0.01},
           'ccffsvm-mp': {1: 0.01},
           'vgg5': {1: 0.1, 60: 0.02, 120: 0.004, 160: 0.0008},
           'vgg5-bn': {1: 0.1, 60: 0.02, 120: 0.004, 160: 0.0008},
           'vgg5-svm': {1: 0.1, 60: 0.02, 120: 0.004, 160: 0.0008},
           'vgg5-bn-svm': {1: 0.1, 60: 0.02, 120: 0.004, 160: 0.0008}}
    curr_lrs = LRs[args.architecture]
    # Get learning rate for the 1st epoch
    lr = curr_lrs[1]
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    if "svm" in args.architecture:
        scores = lasagne.layers.get_output(network, deterministic=True)
        test_loss = network.get_one_vs_all_cost_from_scores(scores, target_var)
        test_prediction = network.get_class_from_scores(scores)
        test_acc = T.mean(T.eq(test_prediction, target_var),
                          dtype=theano.config.floatX)

    else:
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Done building network and compiling functions
    print("Done!")
    print(" # Number of parameters in model: %d" %
          lasagne.layers.count_params(network, trainable=True))

    # Save results
    if args.save_results:
        # Create ./results dir (if doesn't exist)
        results_dir = "./results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Create results file for validation loss and accuracy (over epochs)
        # Filename format:
        # <dataset>_<architecture>_<num_epochs>_<batch_size>_valid.results
        valid_results_filename = "%s/%s_%s_%d_%d_valid_%d.results" % \
                                 (results_dir, args.dataset,
                                  args.architecture, args.num_epochs,
                                  args.batch_size, args.iter)
        if os.path.exists(valid_results_filename):
            os.remove(valid_results_filename)

        # Create results file for test loss and accuracy
        # Filename format:
        # <dataset>_<architecture>_<num_epochs>_<batch_size>_test.results
        test_results_filename = "%s/%s_%s_%d_%d_test_%d.results" % \
                                (results_dir, args.dataset,
                                 args.architecture, args.num_epochs,
                                 args.batch_size, args.iter)
        if os.path.exists(test_results_filename):
            os.remove(test_results_filename)

    # Start training
    print(" # Starting training...")

    # Iterate over epochs:
    for epoch in range(args.num_epochs):

        # Set learning rate
        if epoch in curr_lrs:
            new_lr = curr_lrs[epoch]
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
            if args.verbose:
                print(" # New LR: %f" % sh_lr.get_value())

        # In each epoch, do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # ...and a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, args.batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Print the results for this epoch:
        if args.verbose:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, args.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        if args.save_results:
            with open(valid_results_filename, "a") as valid_results_fid:
                valid_results_fid.write("%.6f %.6f %.6f\n" % (train_err / train_batches,
                                                              val_err / val_batches,
                                                              val_acc / val_batches * 100))

    # Save trained model
    if args.save_model:
        if args.verbose:
            print(" # Save trained model...", end="")
        # Create ./trained_models dir (if doesn't exist)
        trained_models_dir = "./trained_models"
        if not os.path.exists(trained_models_dir):
            os.makedirs(trained_models_dir)
        # Trained model filename format:
        # <dataset>_<architecture>_<num_epochs>_<batch_size>_model.npz
        model_filename = "%s/%s_%s_%d_%d_model_%d.npz" % \
                         (trained_models_dir, args.dataset, args.architecture,
                          args.num_epochs, args.batch_size, args.iter)
        np.savez(model_filename, *lasagne.layers.get_all_param_values(network))
        print("Done!")

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
    print(" # Final results:")
    print("   test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("   test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

    if args.save_results:
        with open(test_results_filename, "w") as test_results_fid:
            test_results_fid.write("%.6f %.6f\n" % (test_err / test_batches,
                                                    test_acc / test_batches * 100))


if __name__ == "__main__":
    main()
