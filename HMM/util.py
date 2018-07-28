#! python
# -*- coding: utf-8 -*-

import random


def sample_discrete_ITM(PMF):
    """
    generate one sample using Inverse Transform Method for a discrete distribution with PMF
    the CDF is [(0, PMF[0]), (1, PMF[0]+PMF[1]), ..., (M-1, PMF[0] + PMF[1] + ... + PMF[M-1])]
    :param PMF: PMF of the target distribution
    :return: the index of the sample
    """
    x = random.random()
    s = 0.0
    for i in range(len(PMF)):
        s += PMF[i]
        if x < s:
            # x = CDF[i] => i = CDF^{-1}(x)
            return i
