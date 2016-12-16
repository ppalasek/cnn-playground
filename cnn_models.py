import lasagne
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import softmax, sigmoid, softplus
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer, dropout, GaussianNoiseLayer, batch_norm

from libuacnn import SVMLayer, SVMGSULayer, ConstantLayer
from libuacnn import addUAInputLayerEst, addUA2DConvLayer, addUAPool2DLayer, addUADenseLayer


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
        svm_input = DenseLayer(incoming=dropout(network, p=0.5),
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')
    else:
        svm_input = DenseLayer(network,
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')

    # Output layer
    network = SVMLayer(svm_input,
                       num_classes=num_classes,
                       sample_dim=256,
                       trainable_C=True,
                       C=15,
                       name='svm')

    return network, [svm_input]


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
    svm_input = DenseLayer(incoming=network,
                         num_units=256,
                         nonlinearity=relu,
                         name='fc')

    # Output layer
    network = SVMLayer(svm_input,
                       num_classes=num_classes,
                       sample_dim=256,
                       trainable_C=True,
                       C=15,
                       name='svm')

    return network, [svm_input]


# ==========================
# Work in the testing branch
# ==========================
def build_ccffsvmgsu(input_var=None, data_shape=None, num_classes=None, pool_mode='average_inc_pad'):

    # UA input layer
    network_mean, network_var = addUAInputLayerEst(shape=data_shape, input_var=input_var, k_0=3)

    # 1st UA convolution layer
    network_mean, network_var = addUA2DConvLayer(network_mean,
                                                 network_var,
                                                 num_filters=64,
                                                 filter_size=(3, 3),
                                                 name="uaconv_1")

    # 1st UA pooling layer
    network_mean, network_var = addUAPool2DLayer(network_mean, network_var,
                                                 pool_size=(2, 2),
                                                 pool_type=pool_mode,
                                                 name='uapool_1')

    # 2nd UA convolution layer
    network_mean, network_var = addUA2DConvLayer(network_mean,
                                                 network_var,
                                                 num_filters=128,
                                                 filter_size=(3, 3),
                                                 name="uaconv_2")
    # 2nd UA pooling layer
    network_mean, network_var = addUAPool2DLayer(network_mean,
                                                 network_var,
                                                 pool_size=(2, 2),
                                                 pool_type=pool_mode,
                                                 name='uapool_2')

    # 1st UA Fully-connected layer
    network_mean, network_var = addUADenseLayer(network_mean,
                                                network_var,
                                                num_units=256,
                                                nonlinearity=relu,
                                                name='ua_fc_1')

    # 2nd Fully-connected layer
    network_mean, network_var = addUADenseLayer(network_mean,
                                                network_var,
                                                num_units=256,
                                                nonlinearity=relu,
                                                name='ua_fc_2')

    # Output layer
    network = SVMGSULayer([network_mean, network_var],
                          num_classes=num_classes,
                          sample_dim=256,
                          trainable_C=True,
                          C=15,
                          name='svm')

    return network, [network_mean, network_var]



def build_ccffsvmgsu_testing(input_var=None, data_shape=None, num_classes=None, pool_mode='average_inc_pad', use_dropout=False):
    """
    With small variances this should behave similar to the network from build_ccffsvm.

    -----------------------
    Architecture: "CCFFSVMgsu-test"
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
        network_mean = DenseLayer(incoming=dropout(network, p=0.5),
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')
    else:
        network_mean = DenseLayer(network,
                             num_units=256,
                             nonlinearity=relu,
                             name='fc_2')


    variances = ConstantLayer(shape=(data_shape[0], 256),
                              value=0.001)

    # Output layer
    network = SVMGSULayer([network_mean, variances],
                          num_classes=num_classes,
                          sample_dim=256,
                          trainable_C=True,
                          C=15,
                          name='svmgsu-testing')

    return network, [network_mean, variances]

