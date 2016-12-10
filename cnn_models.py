import lasagne
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import softmax, sigmoid, softplus
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer, dropout, GaussianNoiseLayer

from layers import SVMlayer as SVMLayer


def build_ccfff_model(input_var=None, data_shape=None, pool_mode='max', use_dropout=False):
    """
    ---------------------
    Architecture: "CCFFF"
    ---------------------
        [Input Layer]
         ConvLayer  : 64 filters, 3x3
         Pooling    : 2x2 ('max', 'average_exc_pad')
         ConvLayer  : 128 filters, 3x3
         Pooling    : 2x2 ('max', 'average_exc_pad')
         FCLayer    : 256 units (+ReLU) (+/- dropout)
         FCLayer    : 256 units (+ReLU) (+/- dropout)
         FCLayer    :  10 units (+Soft-max) = [Output Layer]
    """
    # Input layer
    network = InputLayer(shape=data_shape, input_var=input_var)

    # 1st convolution layer
    network = Conv2DLayer(network,
                          num_filters=64,
                          filter_size=(3, 3),
                          nonlinearity=relu,
                          W=lasagne.init.GlorotUniform(),
                          name='conv_1')
    # 1st pooling layer
    network = Pool2DLayer(network, pool_size=(2, 2), mode=pool_mode, name='pool_1')

    # 2nd convolution layer
    network = Conv2DLayer(network,
                          num_filters=128,
                          filter_size=(3, 3),
                          nonlinearity=relu,
                          W=lasagne.init.GlorotUniform(),
                          name='conv_2')
    # 2nd pooling layer
    network = Pool2DLayer(network, pool_size=(2, 2), mode=pool_mode, name='pool_2')

    # 1st Fully-connected layer
    if use_dropout:
        network = DenseLayer(incoming=dropout(network, p=0.5),
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_1')
    else:
        network = DenseLayer(network,
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_1')

    # 2nd Fully-connected layer
    if use_dropout:
        network = DenseLayer(incoming=dropout(network, p=0.5),
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')
    else:
        network = DenseLayer(network,
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')

    # Output layer
    network = DenseLayer(network,
                         num_units=10,
                         nonlinearity=softmax,
                         name='output')

    return network


def build_ccffsvm_model(input_var=None, data_shape=None, pool_mode='max', use_dropout=False):
    """
    -----------------------
    Architecture: "CCFFSVM"
    -----------------------
        Input Layer
        ConvLayer  : 64 filters, 3x3
        MaxPooling : 2x2 ('max', 'average_exc_pad')
        ConvLayer  : 128 filters, 3x3
        MaxPooling : 2x2 ('max', 'average_exc_pad')
        FCLayer    : 256 units (+ReLU) (+/- dropout)
        FCLayer    : 256 units (+ReLU) (+/- dropout)
        SVMLayer   = [Output Layer]
    """
    # Input layer
    network = InputLayer(shape=data_shape, input_var=input_var)

    # 1st convolution layer
    network = Conv2DLayer(network,
                          num_filters=64,
                          filter_size=(3, 3),
                          nonlinearity=relu,
                          W=lasagne.init.GlorotUniform(),
                          name='conv_1')
    # 1st pooling layer
    network = Pool2DLayer(network, pool_size=(2, 2), mode=pool_mode, name='pool_1')

    # 2nd convolution layer
    network = Conv2DLayer(network,
                          num_filters=128,
                          filter_size=(3, 3),
                          nonlinearity=relu,
                          W=lasagne.init.GlorotUniform(),
                          name='conv_2')
    # 2nd pooling layer
    network = Pool2DLayer(network, pool_size=(2, 2), mode=pool_mode, name='pool_2')

    # 1st Fully-connected layer
    if use_dropout:
        network = DenseLayer(incoming=dropout(network, p=0.5),
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_1')
    else:
        network = DenseLayer(network,
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_1')

    # 2nd Fully-connected layer
    if use_dropout:
        network = DenseLayer(incoming=dropout(network, p=0.5),
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')
    else:
        network = DenseLayer(network,
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')

    # Output layer
    network = SVMLayer(network,
                       return_scores=True,
                       num_classes=10,
                       sample_dim=256,
                       trainable_C=True,
                       C=15,
                       name='svm')

    return network
