from __future__ import print_function
import numpy as np


class SVM_Pegasos():
    '''
    Class has a vector weights in serf.w
    '''
    w__ = []
    max_iter__ = 0
    lambda__ = 1.0

    @property
    def params(self):
        '''
        :param self:
        :return: weight vector and bias
        '''
        return (self.w__)

    def __init__(self, features_size, max_iter, lambda_):
        '''
        :param features_size:
        :param max_iter:
        :param lambda_:
        '''
        self.w__ = np.random.rand(features_size + 1)  # Add bias as first elem
        self.max_iter__ = max_iter
        self.lambda__ = lambda_

    def train(self, data, label):
        '''

        :param data: numpy matrix of datapoints shape=(npoints, nfeat)
        :param label: numpy vector of training classes shape=(npoints)
        :return:
        '''

        # Sample from datapoints max_iter random points
        indexes = np.random.randint(low=0, high=self.w__.shape[0], size=self.max_iter__)
        '''
        numpy.random.randint(low,high=None,size=none,dtype='I')
        return random integers from low(inclusive) to high (exclusive)
        '''
        w_t_next = np.zeros_like(self.w__)

        # Sample from datapoints max_iter random points
        for t in range(1, self.max_iter__):
            datapoint_ = np.hstack((1, data[indexes[t], :]))  # Add bias term
            '''
            numpy.hstack(tup)
            Stack array in sequence horizontally (column wise).
            Take a sequence of arrays and stack them horizontally to make a single array.
            Rebuild arrays divided by hsplit
            '''
            y_t = label[indexes[t]]
            condition = y_t * np.dot(self.w__, datapoint_)
            if condition < 1:
                add_ = self.lambda__ * y_t * datapoint_
                w_t_next = (1 - (1.0 / np.float(t + 1))) * self.w__ + add_
            else:
                w_t_next = (1 - (1.0 / np.float(t + 1))) * self.w__
            self.w__ = w_t_next

    def test(self, datapoints, labels, evaluate_accuracy=True):
        '''

        :param datapoints: test point shape = (npoints,nfeat)
        :param labels: test labels (needed only for accuracy shape = (npoints)
        :param evaluate_accuracy: if True accuracy is returned if false return only results
        :return:
        '''

        acc = None
        results = np.ones(datapoints.shape[0])
        bias_term = np.ones((datapoints.shape[0], 1))
        datapoint_ = np.hstack((bias_term, datapoints))
        sep_plane = np.dot(datapoint_, self.w__)
        results[sep_plane > 0] = 1
        results[sep_plane < 0] = -1
        if (evaluate_accuracy):
            acc = np.sum((results == labels), dtype=np.float) / labels.shape[0]
            return acc, results

        return acc, results
