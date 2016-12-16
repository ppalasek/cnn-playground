import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, GaussianNoiseLayer, Layer, Conv2DLayer, MergeLayer, Pool2DLayer, MaxPool2DLayer, DenseLayer, dropout
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng


def addUAInputLayerEst(shape=None, input_var=None, k_0=3):
    """
    Add an Uncertainty-aware input layer for estimating input
    mean and variance maps from input images
    """
    input_layer = InputLayer(shape=shape, input_var=input_var)
    input_layer_mean = MeanMap(input_layer, k_0)
    input_layer_var = VarianceMap([input_layer, input_layer_mean], k_0)
    input_layer_var = lasagne.layers.ElemwiseMergeLayer([input_layer_mean, input_layer_mean], T.sub)

    return input_layer, input_layer_var


def addUA2DConvLayer(input_layer_mean, input_layer_var,
                     num_filters, filter_size, stride=(1, 1), pad=0):
    """
    """

    # Compute I_{\mu}^{W,b}
    mean_map_W_b = Conv2DLayer(incoming=input_layer_mean,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               pad=pad,
                               W=lasagne.init.GlorotUniform(),
                               name='mean_map_W_b')

    # Compute I_{\Sigma^2}^{W',0}
    var_map_W = Conv2DLayer(incoming=input_layer_var,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            pad=pad,
                            W=T.sqr(mean_map_W_b.W),
                            b=None,
                            name='var_map_W')

    # ----------------------------------------------------------- #
    # So far, we have computed:  *  I_mu^{W,b}                    #
    #                            *  I_\Sigma^{W',0}               #
    # ----------------------------------------------------------- #
    output_layer_mean = computeMu(incomings=[mean_map_W_b, var_map_W])
    output_layer_var = computeSigma2(incomings=[mean_map_W_b, var_map_W])
    output_layer_mean_sq = lasagne.layers.NonlinearityLayer(output_layer_mean, T.sqr)
    output_layer_var = lasagne.layers.ElemwiseMergeLayer([output_layer_var, output_layer_mean_sq], T.sub)

    return output_layer_mean, output_layer_var


def addUAPool2DLayer(input_layer_mean, input_layer_var, pool_size, stride=None, pad=(0, 0), ignore_border=True,
                     pool_type='average', mode='average_inc_pad', **kwargs):
    """
        type = {'max', 'average'}
        mode = {'average_exc_pad', 'average_inc_pad'}
    """
    if pool_type == 'average_inc_pad':
        pooled_layer_mean = Pool2DLayer(input_layer_mean,
                                        pool_size,
                                        stride=stride,
                                        pad=pad,
                                        ignore_border=ignore_border,
                                        mode=mode,
                                        **kwargs)
        pooled_layer_var = Pool2DLayer(input_layer_var,
                                       pool_size,
                                       stride=stride,
                                       pad=pad,
                                       ignore_border=ignore_border,
                                       mode=mode,
                                       **kwargs)
    elif pool_type == 'max':
        # -- Max pool mean --
        pooled_layer_mean = MaxPool2DLayer(input_layer_mean, pool_size)
        pooled_layer_var = MyInverseLayer(input_layer_var, MyInverseLayer(pooled_layer_mean, pooled_layer_mean))

        # -- Max pool variance -- it seems to give worse results.
        # pooled_layer_var = MaxPool2DLayer(input_layer_var, pool_size)
        # pooled_layer_mean = MyInverseLayer(input_layer_mean, MyInverseLayer(pooled_layer_var, pooled_layer_var))

    return pooled_layer_mean, pooled_layer_var


def addUADenseLayer(input_layer_mean, input_layer_var,
                    num_units,
                    W=lasagne.init.GlorotUniform(),
                    b=lasagne.init.Constant(0.),
                    nonlinearity=None,
                    num_leading_axes=1,
                    **kwargs):
    """
    """
    # this is going on in the DenseLayer
    # out = T.dot(input, self.W)
    # and later the bias is added if it's not None
    # out += self.b

    # this should also be fine for calculating the output mean
    # out = b + WX
    # out_mean = b + W * input_layer_mean
    output_layer_mean = DenseLayer(incoming=input_layer_mean,
                                   num_units=num_units,
                                   W=W,
                                   b=b,
                                   nonlinearity=nonlinearity,
                                   num_leading_axes=num_leading_axes,
                                   **kwargs)

    output_layer_var2 = DenseLayerVariance2(incoming=input_layer_var,
                                            num_units=num_units,
                                            W=output_layer_mean.W,
                                            b=None,
                                            nonlinearity=nonlinearity,
                                            num_leading_axes=num_leading_axes,
                                            **kwargs)

    return output_layer_mean, output_layer_var2


class ElemWiseGaussianNoiseLayer(MergeLayer):
    """
    Now you just need to merge this with what's in GaussianNoiseLayer,
    i.e. instantiate self._srng in the constructor, and multiply variances by self._srng.normal(...) if not deterministic.
    """
    def __init__(self, incomings, **kwargs):
        super(ElemWiseGaussianNoiseLayer, self).__init__(incomings, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(*input_shapes))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return output_shape

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input_layer = inputs[0]
        variances = inputs[1]

        if deterministic:
            return input_layer
        else:
            return input_layer + self._srng.normal(input_layer.shape,
                                                   avg=0.0,
                                                   std=variances)


class DenseLayerVariance2(Layer):
    def __init__(self, incoming, num_units, W=None, b=None, nonlinearity=None, num_leading_axes=1, **kwargs):

        super(DenseLayerVariance2, self).__init__(incoming, **kwargs)

        # from lasagne's DenseLayer
        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "leaving no trailing axes for the dot product." %
                (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "requesting more trailing axes than there are input "
                "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                "A DenseLayer requires a fixed input shape (except for "
                "the leading axes). Got %r for num_leading_axes=%d." %
                (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        self.num_units = num_units

        # register the params
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (1, self.num_units)

    def get_output_for(self, input, **kwargs):
        return self.calculate_variance(input)

    def calculate_variance(self, input):
        # the same way as it is done in lasagne's DenseLayer
        if self.num_leading_axes < 0:
            self.num_leading_axes += input.ndim
        if input.ndim > self.num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(self.num_leading_axes + 1)

        # flatten: Returns a view of this tensor with ndim dimensions, whose shape for the first ndim-1 dimensions
        # will be the same as self, and shape in the remaining dimension will be expanded to fit in all the data from self.

        # from http://stackoverflow.com/a/34710378/554606
        # For example, if we flatten a tensor of shape (2, 3, 4, 5) with flatten(x, outdim=2),
        # then we'll have the same (2-1=1) leading dimensions (2,), and the remaining dimensions are collapsed.
        # So the output in this example would have shape (2, 60).

        # so the input is reshaped to (num_images, num_ch * im_h * im_w)

        # this calculating the var for one image:
        def one_step(current_input):
            mul = current_input.dimshuffle((0, 'x')) * self.W
            mul = T.dot(self.W.dimshuffle((1, 0)), mul)

            result_diagonal_indices = T.arange(0, mul.shape[0])

            only_diagonal = mul[result_diagonal_indices, result_diagonal_indices]

            return only_diagonal

        # scan over the first dimension of input which is of shape (num_images, num_ch * im_h * im_w)
        vars_for_all_images, updates = theano.scan(fn=one_step, sequences=[input])

        return vars_for_all_images


# ========================================================
# Experimental (for the implementation of the UA pooling).
# ========================================================
class MyInverseLayer(MergeLayer):
    """

    """
    def __init__(self, incoming, layer, **kwargs):
        super(MyInverseLayer, self).__init__(
            [incoming, layer, getattr(layer, 'input_layer', None) or getattr(layer, 'input_layers', None)[0]], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[2]

    def get_output_for(self, inputs, **kwargs):
        input, layer_out, layer_in = inputs
        return theano.grad(None, wrt=layer_in, known_grads={layer_out: input})


class MeanMap(Layer):
    def __init__(self, incoming, k_0=3, **kwargs):
        super(MeanMap, self).__init__(incoming, **kwargs)

        # we need to pad the input, to simulate the "same" convolution, so
        # I use lasagnas PadLayer here
        pad_width = k_0 // 2
        self.padded_input_layer = lasagne.layers.PadLayer(incoming, width=pad_width)

        W_0 = 2.0 / k_0 ** 2

        self.filter = T.as_tensor_variable(np.ones((1, 1, k_0, k_0)).astype('float32') * W_0)

    def get_output_shape_for(self, input_shape):
        return input_shape  # the output shape is the same as the input shape of our layer

    def get_output_for(self, input, **kwargs):
        return self.calculate_mean(input)  # call the function that does everything

    def calculate_mean(self, input):
        # use the padded input, because theano has only the valid convolution
        padded_input = lasagne.layers.get_output(self.padded_input_layer)

        r = padded_input[:, 0].dimshuffle(0, 'x', 1, 2)
        g = padded_input[:, 1].dimshuffle(0, 'x', 1, 2)
        b = padded_input[:, 2].dimshuffle(0, 'x', 1, 2)

        r_conv = T.nnet.conv.conv2d(r, self.filter, border_mode='valid')
        g_conv = T.nnet.conv.conv2d(g, self.filter, border_mode='valid')
        b_conv = T.nnet.conv.conv2d(b, self.filter, border_mode='valid')

        mean_map = T.concatenate([r_conv, g_conv, b_conv], axis=1)

        return mean_map


class VarianceMap(MergeLayer):
    def __init__(self, incomings, k_0=3, **kwargs):
        super(VarianceMap, self).__init__(incomings, **kwargs)

        W_0 = 1.0 / k_0 ** 2

        self.filter = T.as_tensor_variable(np.ones((1, 1, k_0, k_0)).astype('float32') * W_0)
        self.k_0 = k_0

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]  # the output shape is the same as the input shape of our layer

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mean_map = inputs[1]

        return self.calculate_variance(input, mean_map)

    def calculate_variance(self, input, mean_map):
        input_minus_mean = input - mean_map
        input_layer_var = T.sqr(input_minus_mean)
        input_layer_var /= self.k_0 ** 2

        return input_layer_var


class computeMu(lasagne.layers.base.MergeLayer):
    """
    This layer performs an element-wise merge of a pair input layers using \phi (proposed activation)
    It requires all input layers to have the same output shape.
    Parameters
    ----------
    incomings : a pair of :class:`Layer` instances
                the layers feeding into this layer, or expected input shapes,
                with incoming shapes being equal
    """

    def __init__(self, incomings, **kwargs):
        super(computeMu, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(*input_shapes))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return output_shape

    def get_output_for(self, inputs, **kwargs):
        t_mu = inputs[0]
        # x = inputs[1]
        # y = T.switch(T.eq(x, 0), 10e-12, x)     # if it's 0, replace it with 10e-6, otherwise use x
        # t_sigma = T.sqrt(T.maximum(y, 10e-12))  # use the bigger between 10e-6 and x
        t_sigma = T.sqrt(T.switch(T.eq(inputs[1], 0), 10e-12, inputs[1]))

        rho = t_mu / t_sigma
        erfc = T.erfc(-(1.0 / T.sqrt(2.0)) * rho)
        exp = T.exp(-0.5 * T.sqr(rho))

        output = 0.5 * t_mu * erfc + t_sigma * T.sqrt(1.0 / (2.0 * np.pi)) * exp

        return output


class computeSigma2(lasagne.layers.base.MergeLayer):
    """
    This layer performs an element-wise merge of a pair input layers using \phi (proposed activation)
    It requires all input layers to have the same output shape.
    Parameters
    ----------
    incomings : a pair of :class:`Layer` instances
                the layers feeding into this layer, or expected input shapes,
                with incoming shapes being equal
    """

    def __init__(self, incomings, **kwargs):
        super(computeSigma2, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(*input_shapes))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return output_shape

    def get_output_for(self, inputs, **kwargs):
        t_mu = inputs[0]
        # x = inputs[1]
        # y = T.switch(T.eq(x, 0), 10e-12, x)  # if it's 0, replace it with 10e-6, otherwise use x
        # t_sigma = T.sqrt(T.maximum(y, 10e-12))  # use the bigger between 10e-6 and x
        t_sigma = T.sqrt(T.switch(T.eq(inputs[1], 0), 10e-12, inputs[1]))

        rho = t_mu / t_sigma
        erfc = T.erfc(-(1.0 / T.sqrt(2.0)) * rho)
        exp = T.exp(-0.5 * T.sqr(rho))

        return 0.5 * (T.sqr(t_mu) + T.sqr(t_sigma)) * erfc + T.sqrt(1.0 / (2.0 * np.pi)) * t_mu * t_sigma * exp


class RandomLayer(InputLayer):
    """
        RandomLayer.

        Based on lasagne.layers.InputLayer.
        Instead of giving an input, it always returns a random tensor of given shape.
    """
    def __init__(self, shape, avg=0, std=0.1, random_seed=None, name=None, **kwargs):
        self.shape = shape

        # If no seed is given, use lasagne's random num generator to pick a seed
        if random_seed is None:
            random_seed = lasagne.random.get_rng().randint(1, 2147462579)

        self.srng = RandomStreams(seed=random_seed)
        random_var = self.srng.normal(size=shape, avg=avg, std=std)

        self.input_var = random_var

        if name is None:
            self.name = 'input_random'
        else:
            self.name = name

        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape


class ConstantLayer(InputLayer):
    """
        ConstantLayer.

        Based on lasagne.layers.InputLayer.
        Instead of giving an input, it always returns a constant tensor of given shape.
    """
    def __init__(self, shape, value=0.01, name=None, **kwargs):
        self.shape = shape
        constant = np.ones(shape).astype('float32') * value
        self.input_var = T.as_tensor_variable(constant)

        if name is None:
            self.name = 'input_constant'
        else:
            self.name = name
        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape


def addUAInputLayerRnd(shape=None, input_var=None, batch_size=None):
    """
    Add an Uncertainty-aware input layer for ...
    """
    input_layer = InputLayer(shape=shape, input_var=input_var)
    # Set avg and std if needed, also seed random_seed if needed here
    #input_layer_var = RandomLayer(shape=batch_size, avg=0.1, std=0.01, random_seed=None)
    # ... or use a constant value
    input_layer_var = ConstantLayer(shape=batch_size, value=0.123)
    return input_layer, input_layer_var









class SVMGSULayer(lasagne.layers.MergeLayer):
    """
    This layer implements the SVM-GSU layer.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances
                First input layer are the means, the second layer are the variances
    """

    def __init__(self,
                 incomings,
                 w=lasagne.init.Normal(0.1),
                 b=lasagne.init.Normal(0.1),
                 C=15,
                 trainable_C=True,
                 return_cost=False,
                 targets=None,
                 num_classes=None,
                 sample_dim=None,
                 **kwargs):

        super(SVMGSULayer, self).__init__(incomings, **kwargs)

        assert (w is not None)
        assert (b is not None)
        assert (num_classes is not None)
        assert (sample_dim is not None)

        self.num_classes = num_classes
        self.sample_dim = sample_dim

        self.classes = theano.shared(np.arange(num_classes).astype('int'))

        self.w = self.add_param(w, (self.num_classes, self.sample_dim), name='svm-gsu_w', regularizable=False)
        self.b = self.add_param(b, (self.num_classes,), name='svm-gsu_b', regularizable=False)
        self.C = self.add_param(lasagne.init.Constant(C), (), name='svm-gsu_C', regularizable=False, trainable=trainable_C)

        self.return_cost = return_cost

        self.targets = targets

        if (return_cost):
            assert(self.targets is not None)

    def get_output_shape_for(self, input_shapes):
        if (self.return_cost):
            # input_shapes[0] is the minibatch size of the input
            return (,)
        else:
            return (input_shapes[0][0],)

    def get_output_for(self, inputs, **kwargs):
        if (self.return_cost):
            # the inputs are means, variances
            return self.get_cost(inputs[0], inputs[1], self.targets)
        else:
            return self.classify(inputs[0], inputs[1])


    def get_cost(self, means, variances, targets):
        # THIS IS NOT TESTED YET!

        # target one hot encoded and in {-1, 1}
        t = T.extra_ops.to_one_hot(targets, self.num_classes) * 2
        t -= 1

        # dim of t is [minibatch size, 1]

        # dim of means is [minibatch size, self.sample_dim]
        # dim of self.w [self.num_classes, self.sample_dim]

        # dim of T.dot(means, self.w.T) is [minibatch size, self.num_classes]

        # dim of self.b is [self.num_classes,]

        # dim of d_mu is [minibatch size, self.num_classes]
        d_mu = t - T.dot(means, self.w.T) - self.b

        # adding SQRT_EPS to avoid problems with the derivative of sqrt(x) when
        # x is very small. instead of sqrt(x) we use sqrt(max(x, eps))
        # dim of d_sigma is [minibatch size, self.num_classes]
        d_sigma = T.sqrt(T.maximum(T.dot(variances, self.w.T ** 2)), SQRT_EPS)

        # first part of Equation 5
        erf = 0.5 * d_mu * (T.erf((T.sqrt(2) / 2) * d_mu / d_sigma) + t)

        # second part of Equation 5
        # dim of exp is [minibatch_size, self.num_classes]
        exp = (d_sigma / (T.sqrt(2 * np.pi))) * T.exp(-0.5 * (d_mu / d_sigma) ** 2)

        # regularization
        num_samples = T.cast(target.shape[0], 'float32')
        lambda_coef = 1. / (num_samples * self.C)

        reg = 0.5 * lambda_coef * T.sum(self.w ** 2)

        cost = reg + (erf + exp).mean(axis=0).sum()

        return cost

    def classify(self, means, variances):
        # TODO
        scores = T.dot(means, self.w.T) + self.b

        indices = scores.argmax(axis=1)

        return self.classes[indices]


#
# ==
# ==
#

class SVMLayer(lasagne.layers.Layer):
    def __init__(self,
                 incoming,
                 coef=lasagne.init.Normal(0.1),
                 intercept=lasagne.init.Normal(0.1),
                 C=15,
                 trainable_C=True,
                 return_scores=False,
                 num_classes=None,
                 sample_dim=None,
                 **kwargs):

        super(SVMLayer, self).__init__(incoming, **kwargs)

        assert (coef is not None)
        assert (intercept is not None)
        assert (num_classes is not None)
        assert (sample_dim is not None)

        self.num_classes = num_classes
        self.sample_dim = sample_dim

        self.classes = theano.shared(np.arange(num_classes).astype('int'))

        # the regularization is already explicitly added later
        self._coef = self.add_param(coef, (self.num_classes, self.sample_dim), name='svm_coef', regularizable=False)
        self._intercept = self.add_param(intercept, (self.num_classes,), name='svm_intercept', regularizable=False)
        self.C = self.add_param(lasagne.init.Constant(C), (), name='svm_C', regularizable=False, trainable=trainable_C)

        self.return_scores = return_scores

    def get_output_shape_for(self, input_shape):
        if (self.return_scores):
            return (input_shape[0], self.num_classes)
        else:
            return (input_shape[0],)

    def get_output_for(self, input, **kwargs):
        if (self.return_scores):
            return self.get_scores(input)
        else:
            return self.classify(input)

    def get_scores(self, sample):
        scores = T.dot(sample, self._coef.T) + self._intercept

        return scores

    def get_class_from_scores(self, scores):
        indices = scores.argmax(axis=1)

        return self.classes[indices]

    def classify(self, sample):
        scores = self.get_scores(sample)

        return get_class_from_scores(scores)

    def get_one_vs_all_cost_from_scores(self, scores, target):
        # this is squared hinge loss!

        # target one hot encoded and in {-1, 1}
        y_i = T.extra_ops.to_one_hot(target, self.num_classes) * 2
        y_i -= 1

        num_samples = T.cast(target.shape[0], 'float32')
        lambda_coef = 1. / (num_samples * self.C)

        cost = T.maximum(0, 1 - y_i * scores) ** 2
        final_cost = cost.mean(axis=0).sum()
        final_cost += 0.5 * lambda_coef * T.sum(self._coef ** 2)

        return final_cost