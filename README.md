#cnn-playground

A Convolutional Neural Network (CNN) playground using Theano/Lasagne.

This framework supports the following datasets:

- CIFAR-10 (see https://www.cs.toronto.edu/~kriz/cifar.html)
- CIFAR-100 (see https://www.cs.toronto.edu/~kriz/cifar.html)
- MNIST (see http://yann.lecun.com/exdb/mnist/)
- SVHN (see http://ufldl.stanford.edu/housenumbers/)

and a number of basic architectures for training and evaluating a CNN. More specifically, the following architecture are available:

~~~
	 'CCFF':
~~~


## Step 1: Get the datasets

First, you need to populate the `data/` directory as follows:


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

To this end, you need to run the `get_data.py` script and select which dataset(s) you want to download. If you already have the datasets, just copy them in the appropriate directories/subdirectories, as shown above and skip running `get_data.py`.


## Step 2: Train a CNN

For training a CNN, you need to run the `trainCNN.py` script. It's usage is shown below:

~~~
runCNN.py <arguments>
~~~

where

* `A`
* `B`







