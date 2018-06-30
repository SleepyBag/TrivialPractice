from numpy import *
import numpy


def regression(x, y, traverse=False):
    x = array(x)
    if traverse == True:
        x = x.T

    ones = numpy.ones(shape=(x.shape[0], 1))
    x = numpy.concatenate((ones, x), axis=1)
    y = array([[i] for i in y])
    b = ndarray.dot(numpy.linalg.inv(ndarray.dot(x.T, x)), ndarray.dot(x.T, y))
    return b
