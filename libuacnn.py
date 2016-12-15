import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, GaussianNoiseLayer, Layer, Conv2DLayer, MergeLayer, Pool2DLayer, MaxPool2DLayer, DenseLayer, dropout
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng

