#! python
# -*- coding: utf-8 -*-

import numpy as np
import random

"""
    A Hidden Markov Model is represented by a five-tuple (Q, V, A, B, P), where
    Q = {q_0, q_1, ..., q_{M-1}} is the set of M hidden states;
    V = {v_0, v_1, ..., v_{N-1}} is the set of N observed states;
    A = [a_mk] (0 <= m, k <= M-1) is a M x M matrix, denoting transition probability between the hidden states;
    B = [b_mn] (0 <= m <= M-1, 0 <= n <= N-1) is a M x N matrix, denoting the generation probability between Q and V;
    P = [p_m] (0 <= m <= M-1) is the initial distribution of the hidden states at time t=0
    Actually, we don't care the content of Q and V, only M and N are necessary. 
"""


def sample_ITMD(PMF):
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


class HMM:
    def __init__(self, A=[], B=[], P=[]):
        if (np.size(A, 0) != np.size((A, 1))) or (np.size(A, 0) != np.size(B, 0)) or (np.size(A, 0) != np.size(P)):
            print(np.size(A, 0), np.size(A, 1), np.size(B, 0), np.size(P))
            print('The parameters are not consistent...')
            # exit(-1)
        self.M = np.size(B, 0)
        self.N = np.size(B, 1)
        self.A = np.array(A)
        self.B = np.array(B)
        self.P = np.array(P)

    def generation(self, T):
        """
        :param T: number of samples
        :return I: list of T indices, denoting T samples of the hidden states {i_0, i_1, ..., i_{T-1}}
        :return O: list of T indices, denoting T samples of the observed states {o_0, o_1, ..., o_{T-1}}
        """
        I, O = [], []
        if self.A is None or self.B is None or self.P is None:
            print('The parameters are not given, I cannot generate the samples')
            return I, O
        # generate the first hidden state i_0 according to P
        I.append(sample_ITMD(self.P))
        # generate the states at time 1 to T-1
        for t in range(1, T):
            i = I[-1]
            O.append(sample_ITMD(self.B[i, :]))
            I.append(sample_ITMD(self.A[i, :]))
        O.append(sample_ITMD(self.B[i, :]))
        return I, O

    def evaluate_forward(self, O):
        """
        ##

          Given the model and observed samples O, evaluate the probability that the model generating O

        ##

        Define the forward probability:
          f_m(t) = Pr(o_0, o_1, ..., o_t, i_t = q_m | A, B, P), where 0 <= m <= M-1, 0 <= t <= T-1
        then:
          f_m(t) = (sum_{k=1}^{M} f_k(t-1) * a_km) * b_m(o_t), where 0 <= m <= M-1, 1 <= t <= T-1.

        Note: f_m(0) = ...

        ##

        Let f(t) = [f_0(t), f_1(t), ..., f_{M-1}(t)], a row vector, then we have the equation in matrix form:
          f(t) = (f(t-1) . A) * B[:, o_t] for any 1 <= t <= T-1  --- (eq.1),
        where "." is matrix multiplication (i.e., "." <=> np.dot())
              "*" multiplies arguments element-wise (i.e., "*" <=> np.multiply())

        ##

        :param O:
        :return: Pr(O|A, B, P)
        """
        T = len(O)
        # forward_prob will be a T x M matrix
        # the (t, m)-th entry is the forward probability f_m(t)
        forward_prob = []

        # for np.array: * multiplies arguments element-wise.
        # => row vector
        f0 = self.P * self.B[:, O[0]]
        forward_prob.append(f0)
        for t in range(1, T):
            # f(t-1)
            ft_1 = forward_prob[-1]
            # eq.1, matrix form
            ft = np.dot(ft_1, self.A) * self.B[:, O[t]]
            forward_prob.append(ft)
        return np.sum(forward_prob[-1])

    def evaluate_backward(self, O):
        """
        ##

          Given the model and observed samples O, evaluate the probability that the model generating O

        ##

        Define the backward probability:
          c_m(t) = Pr(o_{t+1}, o_{t+2}, ..., o_{T-1} | i_t = q_m , A, B, P), where 0 <= m <= M-1, 0 <= t <= T-1
        then:
          c_m(t) = sum_{k=1}^{M} (c_k(t+1) * b_k(o_t) * a_mk, where 0 <= m <= M-1, 0 <= t <= T-2.

        Note: c_m(T-1) = 1 for any 0 <= m <= M-1

        ##

        Let c(t) = [c_1(t), c_2(t), ..., c_M(t)], a row vector, then we have the equation in matrix form:
          c(t) = A . (c(t+1) * B[:, o_t]), for any 0 <= t <= T-2  --- (eq.2),
        where "A'" is the transition of matrix A
              "." is matrix multiplication (i.e., "." <=> np.dot())
              "*" multiplies arguments element-wise (i.e., "*" <=> np.multiply())

        ##

        :param O:
        :return: Pr(O|A, B, P)
        """
        T = len(O)
        # backward_prob will be a T x M matrix
        # the (t, m)-th entry is the forward probability c_m(T-1-t)
        backward_prob = []

        # for t = T-1, c_m(t) = 1
        backward_prob.append([1] * self.M)
        for t in range(T-2, -1, -1):
            # c(t+1)
            c_t1 = backward_prob[-1]
            c_t = np.dot(self.A, c_t1 * self.B[:, O[t+1]])
            backward_prob.append(c_t)
        return np.sum(backward_prob[-1] * self.B[:, O[0]] * self.P)


    def decoding_Vertebi(self, O):
        """
        Given the model and observed samples O, find the most likely hidden states I that generates O
        :param O:
        :return I: The most likely hidden state samples I = arg max Pr(I|O; A, B, P)
        """
        pass

    def learning_EM(self, O):
        """
        learn the parameter of the model using Baum-Welch (i.e., EM) algorithm
        :param O:
        :return A, B, P: (A, B, P) = arg max Pr(O|A, B, P)
        """
        pass

    def learning_supervised(self, I, O):
        """
        learn the parameter of the model using supervised method
        :param I:
        :param O:
        :return A, B, P: (A, B, P) = arg max Pr(I, O|A, B, P)
        """
        pass


if '__main__' == __name__:
    A = np.array([(0.5, 0.2, 0.3), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5)])
    B = np.array([(0.5, 0.5), (0.4, 0.6), (0.7, 0.3)])
    P = np.array((0.2, 0.4, 0.4))
    out_samples = (0, 1, 0)

    hmm_instance = HMM(A, B, P)
    # print(hmm_instance.generation(100))
    print(hmm_instance.evaluate_forward(out_samples))
    print(hmm_instance.evaluate_backward(out_samples))