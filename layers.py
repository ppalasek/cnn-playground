import theano
import numpy as np

import theano.tensor as T

import lasagne

class SVMlayer(lasagne.layers.Layer):
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

        super(SVMlayer, self).__init__(incoming, **kwargs)

        assert(coef is not None)
        assert(intercept is not None)
        assert(num_classes is not None)
        assert(sample_dim is not None)

        self.num_classes = num_classes
        self.sample_dim = sample_dim

        self.classes = theano.shared(np.arange(num_classes).astype('int'))

        # the regularization is already explicitly added later
        self._coef = self.add_param(coef, (self.num_classes, self.sample_dim), name='svm_coef', regularizable=False)
        self._intercept = self.add_param(intercept, (self.num_classes, ), name='svm_intercept', regularizable=False)
        self.C = self.add_param(lasagne.init.Constant(C), (), name='svm_C', regularizable=False, trainable=trainable_C)

        self.return_scores = return_scores

    def get_output_shape_for(self, input_shape):
        if (self.return_scores):
            return (input_shape[0], self.num_classes)
        else:
            return (input_shape[0], )

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

        cost =  T.maximum(0, 1 - y_i * scores) ** 2
        final_cost = cost.mean(axis=0).sum()
        final_cost += 0.5 * lambda_coef * T.sum(self._coef ** 2)

        return final_cost
