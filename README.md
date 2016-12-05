#cnn-playground

A framework for training/testing a Convolutional Neural Network (CNN) using Theano/Lasagne.

###Prerequisites

 -- Theano (Prefer the [bleeding-edge version](http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions))
 -- Lasagne (Prefer the [bleeding-edge version](http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version))
 -- Install the latest CUDA Toolkit and possibly the corresponding driver available from NVIDIA: https://developer.nvidia.com/cuda-downloads

To configure Theano to *use the GPU by default*, create a file `.theanorc` directly in your home directory, with the following contents:
~~~
[global]
floatX = float32
device = gpu
~~~

Optionally add `allow_gc = False` for some extra performance at the expense of (sometimes substantially) higher GPU memory usage.


###Description

This framework supports the following **datasets**:

- CIFAR-10 (see https://www.cs.toronto.edu/~kriz/cifar.html)
- CIFAR-100 (see https://www.cs.toronto.edu/~kriz/cifar.html)
- MNIST (see http://yann.lecun.com/exdb/mnist/)
- SVHN (see http://ufldl.stanford.edu/housenumbers/)

and a set of basic **architectures** for training and evaluating a CNN. More specifically, the following architectures are available:


 - **`'ccfff'`**: A simple architecture consisting of 2 convolution and 3 fully-connected layers (see `build_ccfff_model()` in `cnn_models.py` for more details)
 - *More to be added.*



#### Step 1: Get the datasets

First, you need to populate the `./data/` directory as follows:


		./data/
		├── cifar10
		│   ├── batches.meta
		│   ├── data_batch_1
		│   ├── data_batch_2
		│   ├── data_batch_3
		│   ├── data_batch_4
		│   ├── data_batch_5
		│   ├── readme.html
		│   └── test_batch
		├── cifar100
		│   ├── meta
		│   ├── test
		│   └── train
		├── mnist
		│   ├── t10k-images-idx3-ubyte
		│   ├── t10k-labels-idx1-ubyte
		│   ├── train-images-idx3-ubyte
		│   └── train-labels-idx1-ubyte
		└── svhn
		    ├── test_32x32.mat
		    └── train_32x32.mat

To this end, you need to run the `get_data.py` script and select which dataset(s) you want to download. If you already have the datasets (in the above form), just copy them in the appropriate directories/subdirectories, as shown above and skip running `get_data.py`.



#### Step 2: Train and evaluate a CNN

For training/evaluating a CNN, you need to run the `runCNN.py` script. It's usage is shown below (also, you may run `python runCNN.py -h` for detailed ):

Usage:
~~~
runCNN.py [-h][-v][-d <DATASET>][-a <ARCHITECTURE>][-e NUM_EPOCHS][-b BATCH_SIZE][-l <LOSS>][-s]
~~~
Arguments:
~~~
	-h, --help : show help message
	-v, --verbose : increase output verbosity
	-d, --dataset : choose dataset from {'mnist', 'cifar10', 'cifar100', 'svhn'} (default: 'cifar10')
	-a, --architecture : choose architecture from {'ccfff'} (default: 'ccfff')
	-e, --num_epochs : set number of epochs
	-b, --batch_size : set batch size
	-l, --loss : choose loss function
	-s, --save_model : save model file
~~~
