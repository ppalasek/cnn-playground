import lasagne
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import softmax, sigmoid, softplus
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer, dropout, GaussianNoiseLayer, batch_norm

from libuacnn import SVMLayer
from libuacnn import SVMGSULayer
from libuacnn import ConstantLayer


def build_ccfff(input_var=None, data_shape=None, num_classes=None, pool_mode='average_inc_pad', use_dropout=False):
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
                         num_units=num_classes,
                         nonlinearity=softmax,
                         name='output')

    return network


def build_ccffsvm(input_var=None, data_shape=None, num_classes=None, pool_mode='average_inc_pad', use_dropout=False):
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
                       num_classes=num_classes,
                       sample_dim=256,
                       trainable_C=True,
                       C=15,
                       name='svm')

    return network


def build_ccffsvmgsu(input_var=None, data_shape=None, num_classes=None, target_var=None, pool_mode='average_inc_pad', use_dropout=False):
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
        SVMGSULayer   = [Output Layer]
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


    variances = ConstantLayer(shape=(data_shape[0], 256),
                              value=0.001)

    # Output layer
    network = SVMGSULayer([network, variances],
                       return_cost=target_var is not None,
                       targets=target_var,
                       num_classes=num_classes,
                       sample_dim=256,
                       trainable_C=True,
                       C=15,
                       name='svmgsu')

    return network




def build_vgg5(input_var=None, data_shape=None, num_classes=None, do_batch_norm=False):
    """

    """
    network = InputLayer(shape=data_shape, input_var=input_var)

    # 1st convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=64,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_1'))
    else:
        network = Conv2DLayer(network,
                              num_filters=64,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_1')

    # 1st pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_1')

    # 2nd convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=128,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_2'))
    else:
        network = Conv2DLayer(network,
                              num_filters=128,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_2')

    # 2nd pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_2')

    # 3rd convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=256,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_3'))
    else:
        network = Conv2DLayer(network,
                              num_filters=256,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_3')

    # 3rd pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_3')

    # 4th convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=512,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_4'))
    else:
        network = Conv2DLayer(network,
                              num_filters=512,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_4')

    # 4th pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_4')

    # 5th convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=512,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_5'))
    else:
        network = Conv2DLayer(network,
                              num_filters=512,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_4')

    # 5th pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), mode='average_inc_pad', name='pool_5')

    # Fully-connected layer with 10 units
    network = DenseLayer(incoming=network,
                         num_units=num_classes,
                         nonlinearity=softmax,
                         name='output')

    return network


def build_vgg5_svm(input_var=None, data_shape=None, num_classes=None, do_batch_norm=False):
    """

    """
    network = InputLayer(shape=data_shape, input_var=input_var)

    # 1st convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=64,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_1'))
    else:
        network = Conv2DLayer(network,
                              num_filters=64,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_1')

    # 1st pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_1')

    # 2nd convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=128,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_2'))
    else:
        network = Conv2DLayer(network,
                              num_filters=128,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_2')

    # 2nd pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_2')

    # 3rd convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=256,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_3'))
    else:
        network = Conv2DLayer(network,
                              num_filters=256,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_3')

    # 3rd pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_3')

    # 4th convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=512,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_4'))
    else:
        network = Conv2DLayer(network,
                              num_filters=512,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_4')

    # 4th pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), name='pool_4')

    # 5th convolution layer
    if do_batch_norm:
        network = batch_norm(Conv2DLayer(network,
                                         num_filters=512,
                                         filter_size=(3, 3),
                                         pad=1,
                                         flip_filters=False,
                                         name='conv_5'))
    else:
        network = Conv2DLayer(network,
                              num_filters=512,
                              filter_size=(3, 3),
                              pad=1,
                              flip_filters=False,
                              name='conv_4')

    # 5th pooling layer (max-pooling)
    network = Pool2DLayer(network, pool_size=(2, 2), mode='average_inc_pad', name='pool_5')

    # Fully-connected layer
    network = DenseLayer(incoming=network,
                         num_units=256,
                         nonlinearity=relu,
                         name='fc')

    # Output layer
    network = SVMLayer(network,
                       return_scores=True,
                       num_classes=num_classes,
                       sample_dim=256,
                       trainable_C=True,
                       C=15,
                       name='svm')

    return network