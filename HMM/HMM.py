#! python
# -*- coding: utf-8 -*-

from util import *
import numpy as np

"""
    A Hidden Markov Model is represented by a five-tuple (Q, V, A, B, P), where
    Q = {q_0, q_1, ..., q_{M-1}} is the set of M hidden states;
    V = {v_0, v_1, ..., v_{N-1}} is the set of N observed states;
    A = [a_mk] (0 <= m, k <= M-1) is a M x M matrix, denoting transition probability between the hidden states;
    B = [b_mn] (0 <= m <= M-1, 0 <= n <= N-1) is a M x N matrix, denoting the generation probability between Q and V;
    P = [p_m] (0 <= m <= M-1) is the initial distribution of the hidden states at time t=0
    Actually, we don't care the content of Q and V, only M and N are necessary. 
"""


class HMM:
    def __init__(self, A=[], B=[], P=[]):
        if (np.size(A, 0) != np.size(A, 1)) or (np.size(A, 0) != np.size(B, 0)) or (np.size(A, 0) != np.size(P)):
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
        I.append(sample_discrete_ITM(self.P))
        # generate the states at time 1 to T-1
        for t in range(1, T):
            i = I[-1]
            O.append(sample_discrete_ITM(self.B[i, :]))
            I.append(sample_discrete_ITM(self.A[i, :]))
        O.append(sample_discrete_ITM(self.B[i, :]))
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

        Let c(t) = [c_0(t), c_1(t), ..., c_{M-1}(t)], a row vector, then we have the equation in matrix form:
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

        ##

        Let
          delta_t(m) = max_{i_0, ..., i_{t-1}}Pr(i_0, ..., i_t = q_m, o_1, ..., o_t | A, B, P), 0 <= t <= T-1, 0 <= m <= M-1
        Then
          delta_t(m) = max_{0 <= k <= M-1}[Pr(i_t = q_m, o_t | i_{t-1} = q_k, A, B, P)
                                           * max_{i_0, ..., i_{t-2}}Pr(i_0, ..., i_{t-1} = q_k, o_1, ..., o_{t-1} | A, B, P)]
                     = max_{0 <= k <= M-1}[Pr(i_t = q_m, o_t | i_{t-1} = q_k, A, B, P) * delta_{t-1}(k)],
                     = max_{0 <= k <= M-1}[delta_{t-1}(k) * a_km * b_m(o_t)], 1 <= t <= T-1, 0 <= m <= M-1
        Let
          H_t(m) = arg max_{0 <= k <= M-1} [delta_{t-1}(k) * a_km] = arg max_{0 <= k <= M-1} [delta_{t-1}(k) * a_km * b_m(o_t)]
          H_0(m) = None, for 0 <= m <= M-1

        ##

        Let delta_t = (delta_t(0), delta_t(1), ..., delta_t(M-1)), a row vector.
        We multiply the k-th row of A by delta_{t-1}(k), multiply the m-th column by b_m(o_t),
        and then the maximum in each column is delta_t(m).

        ##

        :param O:
        :return I: The most likely hidden state samples I = arg max Pr(I|O; A, B, P)
        """
        T = len(O)

        # delta will be a T x M matrix
        # the (t, m)-th entry is delta_t(m)
        delta = []
        H = []

        # for t = 0
        delta.append(self.P * self.B[:, O[0]])
        H.append([None] * self.M)
        # for t = 1, 2, ..., T-1
        for t in range(1, T):
            # delta_{t-1}
            delta_t1 = delta[-1]
            # A的第k行乘以delta_{t-1}(k)
            delta_t = np.array([delta_t1[k] * self.A[k, :] for k in range(0, self.M)])
            # 第m列乘以b_m(o_t)，注意要取转置，因为[column for m in ...]相当于转置一次
            delta_t = np.array([B[m, O[t]] * delta_t[:, m] for m in range(0, self.M)]).transpose()
            # 取各列的最大元素值
            # 以及各列中最大元素所在的位置k*，即到达t时刻状态q_m，需要经过上一个时刻的状态q_k*，i.e. k* = arg max_{k} delta_t(m)
            delta_t, h = np.amax(delta_t, axis=0), np.argmax(delta_t, axis=0)
            delta.append(delta_t)
            H.append(h)
        # 求T-1时刻使得delta_{T-1}最大的状态q_m*, i.e. m* = arg max_{m} delta_{T-1}(m)
        I = [np.argmax(delta[T-1])]
        for t in range(T-2, -1, -1):  # t = T-2, T-1, ..., 0
            I.append(H[t+1][I[-1]])
        I.reverse()
        return I


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
    print(hmm_instance.decoding_Vertebi(out_samples))