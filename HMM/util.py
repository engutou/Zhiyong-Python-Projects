#! python
# -*- coding: utf-8 -*-

import random
import numpy as np
from scipy.stats import rv_discrete


def sample_discrete_ITM(PMF, size=1):
    """
    generate one sample using Inverse Transform Method for a discrete distribution with PMF
    the CDF is [(0, PMF[0]), (1, PMF[0]+PMF[1]), ..., (M-1, PMF[0] + PMF[1] + ... + PMF[M-1])]
    :param PMF: PMF of the target distribution
    :return: the index of the sample
    """
    x_list = [random.random() for i in range(size)]
    ret = []
    for x in x_list:
        s = 0.0
        for i in range(len(PMF)):
            s += PMF[i]
            if x < s:
                # x = CDF[i] => i = CDF^{-1}(x)
                ret.append(i)
                break
    return ret


def sample_discrete_rv(PMF, size=1):
    """
    same as sample_discrete_ITM(PMF, size)
    If size is large, use this one
    If need to repeat a lot of times, use the other one
    :param PMF:
    :param size:
    :return:
    """
    return rv_discrete(values=(range(len(PMF)), PMF)).rvs(size=size)


def is_equal_iterable(i1, i2):
    if len(i1) == len(i2):
        return all(is_equal(i1[k], i2[k]) for k in range(len(i1)))
    else:
        return False


def is_equal(f1, f2):
    return abs(f1 - f2) <= 1e-8


######################################################################################################
import time
if '__main__' == __name__:
    pmf = (0.2, 0.4, 0.4)

    t = time.time()
    samples = []
    for i in range(1000000):
        samples.extend(sample_discrete_ITM(pmf))
    print(np.mean(samples), np.var(samples), time.time() - t)

    t = time.time()
    samples = sample_discrete_ITM(pmf, size=1000000)
    print(np.mean(samples), np.var(samples), time.time() - t)

    t = time.time()
    samples = []
    # very slow
    for i in range(10000):
        samples.append(sample_discrete_rv(pmf))
    print(np.mean(samples),  np.var(samples), time.time() - t)

    t = time.time()
    samples = sample_discrete_rv(pmf, size=1000000)
    print(np.mean(samples), np.var(samples), time.time() - t)
