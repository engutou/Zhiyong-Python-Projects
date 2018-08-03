#! python
# -*- coding: utf-8 -*-

from util import *
import numpy as np

"""
    todo: 验证generation的正确性
          完成学习算法本身
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
        I.append(sample_discrete_ITM(self.P)[0])
        # generate the states at time 1 to T-1
        for t in range(1, T):
            i = I[-1]
            O.append(sample_discrete_ITM(self.B[i, :])[0])
            I.append(sample_discrete_ITM(self.A[i, :])[0])
        O.append(sample_discrete_ITM(self.B[i, :])[0])
        return I, O

    def get_forward_prob(self, O):
        """
        ##

        Given the model (A, B, P) and observed samples O, compute the forward probability

        ##

        Define the forward probability:
          f_t(m) = Pr(o_0, o_1, ..., o_t, i_t = q_m | A, B, P), where 0 <= m <= M-1, 0 <= t <= T-1
        then:
          f_t(m) = (sum_{k=1}^{M} f_{t-1}(k) * a_km) * b_m(o_t), where 0 <= m <= M-1, 1 <= t <= T-1.

        Note: f_0(m) = p_m * b_m(o_0) for any 0 <= m <= M-1

        ##

        Let f_t = [f_t(0), f_t(1), ..., f_t(M-1)], a row vector, then we have the equation in matrix form:
            f_t = (f_{t-1} . A) * B[:, o_t] for any 1 <= t <= T-1  --- (eq.1),
        where "." is matrix multiplication (i.e., "." <=> np.dot())
              "*" multiplies arguments element-wise (i.e., "*" <=> np.multiply())

        ##

        :param O:
        :return: Pr(O|A, B, P)
        """
        T = len(O)
        # forward_prob will be a T x M matrix
        # the (t, m)-th entry is the forward probability f_t(m)
        forward_prob = []

        # for np.array: * multiplies arguments element-wise.
        # => row vector
        f0 = self.P * self.B[:, O[0]]
        forward_prob.append(f0)
        for t in range(1, T):
            # ft_1 <=> f_{t-1}
            ft_1 = forward_prob[-1]
            # eq.1, matrix form
            ft = np.dot(ft_1, self.A) * self.B[:, O[t]]
            forward_prob.append(ft)
        return np.array(forward_prob)

    def get_backward_prob(self, O):
        """
        ##

        Given the model (A, B, P) and observed samples O, compute the forward probability

        ##

        Define the backward probability:
          c_t(m) = Pr(o_{t+1}, o_{t+2}, ..., o_{T-1} | i_t = q_m , A, B, P), where 0 <= m <= M-1, 0 <= t <= T-1
        then:
          c_t(m) = sum_{k=1}^{M} (c_{t+1}(k) * b_k(o_t) * a_mk, where 0 <= m <= M-1, 0 <= t <= T-2.

        Note: c_{T-1}(m) = 1 for any 0 <= m <= M-1

        ##

        Let c_t = [c_t(0), c_t(1), ..., c_t(M-1)], a row vector, then we have the equation in matrix form:
            c_t = A . (c_{t+1} * B[:, o_t]), for any 0 <= t <= T-2  --- (eq.2),
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

        # for t = T-1, c_t(m) = 1
        backward_prob.append(np.array([1] * self.M))
        for t in range(T-2, -1, -1):
            # c_t1 <=> c{t+1}
            c_t1 = backward_prob[-1]
            c_t = np.dot(self.A, c_t1 * self.B[:, O[t+1]])
            backward_prob.append(c_t)
        backward_prob.reverse()
        return np.array(backward_prob)

    def verify_forward_backward(self, O):
        """
        If we get the forward probability f_t(m)and the backward probability c_t(m) for any t and m, then
        Pr(O | A, B, P) can be evaluated by Pr(O | A, B, P) = sum_{m=0}^{M-1} c_t(m) * f_t(m) with any t!
        :param O:
        :return:
        """
        forward_prob = self.get_forward_prob(O)
        backward_prob = self.get_backward_prob(O)
        print('Verify forward and backward:', np.sum(forward_prob * backward_prob, axis=1))

    def get_single_state_prob(self, O, forward_prob=None, backward_prob=None):
        """
        ##

          Given the model (A, B, P) and the observed samples O,
          evaluate the distribution of a single hidden state at each time,
          i.e. Pr(i_t = q_m | O, A, B, P) for any 0 <= m <= M-1 and 0 <= t <= T-1

        ##

        Let gamma_t(m) = Pr(i_t = q_m | O, A, B, P)
                       = Pr(i_t = q_m, O | A, B, P) / Pr(O | A, B, P).
        As Pr(i_t = q_m, O | A, B, P) = Pr(o_1, o_2, ..., o_t, i_t = q_m, o_{t+1}, ..., o_{T-1} | A, B, P)
                                      = Pr(o_{t+1}, ..., o_{T-1} | o_1, o_2, ..., o_t, i_t = q_m, A, B, P) * Pr(o_1, o_2, ..., o_t, i_t = q_m | A, B, P)
                                      = Pr(o_{t+1}, ..., o_{T-1} | i_t = q_m, A, B, P) * f_t(m)
                                      = c_t(m) * f_t(m)
        and Pr(O | A, B, P) = sum_{m=0}^{M-1} Pr(i_t = q_m, O | A, B, P) = sum_{m=0}^{M-1} c_t(m) * f_t(m),
        we have gamma_t(m) = c_t(m) * f_t(m) / (sum_{m=0}^{M-1} c_t(m) * f_t(m))

        ##

            Extra note: If we get the forward probability f_t(m)and the backward probability c_t(m) for any t and m, then
                  Pr(O | A, B, P) can be evaluated by Pr(O | A, B, P) = sum_{m=0}^{M-1} c_t(m) * f_t(m) with any t!

                  For t = 0:
                      f_0(m) = p_m * b_m(o_0) for any 0 <= m <= M-1,
                      we have Pr(O | A, B, P) = sum_{m=0}^{M-1} c_0(m) * b_m(o_0) * f_0(m)  ==>  evaluate_backward()

                  For t = T-1:
                      c_{T-1}(m) = 1 for any 0 <= m <= M-1,
                      we have Pr(O | A, B, P) = sum_{m=0}^{M-1} f_{T-1}(m)  ==>  evaluate_forward()

        ##
        :return:
        """
        if not forward_prob:
            forward_prob = self.get_forward_prob(O)
        if not backward_prob:
            backward_prob = self.get_backward_prob(O)
        # "*" is element-wise multiplication
        Pr_O = np.sum(forward_prob[-1])
        return forward_prob * backward_prob / Pr_O

    def get_double_state_prob(self, O, forward_prob=None, backward_prob=None):
        """
        ##

          Given the model (A, B, P) and the observed samples O,
          evaluate the distribution of two hidden states at time t and t+1,
          i.e. Pr(i_t = q_m, i_{t+1} = q_k | O, A, B, P) for any 0 <= m, k <= M-1 and 0 <= t <= T-2

        ##

        Let xi_t(m, k) = Pr(i_t = q_m, i_{t+1} = q_k | O, A, B, P)
                       = Pr(i_t = q_m, i_{t+1} = q_k, O | A, B, P) / Pr(O | A, B, P).
        As Pr(i_t = q_m, i_{t+1} = q_k, O | A, B, P) = Pr(o_1, o_2, ..., o_t, i_t = q_m, i_{t+1} = q_k, o_{t+1}, ..., o_{T-1} | A, B, P)
                                      = Pr(o_{t+2}, ..., o_{T-1} | i_{t+1} = q_k, A, B, P) * Pr(o_1, o_2, ..., o_t, i_t = q_m, i_{t+1} = q_k, o_{t+1} | A, B, P)
                                      = c_{t+1}(k) * Pr(o_{t+1} | i_{t+1} = q_k, A, B, P) * Pr(o_1, o_2, ..., o_t, i_t = q_m, i_{t+1} = q_k | A, B, P)
                                      = c_{t+1}(k) * b_k(o_{t+1}) * Pr(i_{t+1} = q_k | i_t = q_m, A, B, P) * Pr(o_1, o_2, ..., o_t, i_t = q_m | A, B, P)
                                      = c_{t+1}(k) * b_k(o_{t+1}) * a_mk * f_t(m)
                                      = f_t(m) * a_mk * b_k(o_{t+1}) * c_{t+1}(k)
        we have xi_t(m, k) = f_t(m) * a_mk * b_k(o_{t+1}) * c_{t+1}(m) / Pr(O | A, B, P)

        ##
        :return:
        """
        if not forward_prob:
            forward_prob = self.get_forward_prob(O)
        if not backward_prob:
            backward_prob = self.get_backward_prob(O)

        T = len(O)
        double_state_prob = []
        for t in range(T-1):
            double_state_prob_t = np.array([forward_prob[t, m] * self.A[m, :] for m in range(self.M)])
            # 按列操作后放到list里面，需要翻转
            double_state_prob_t = np.array([double_state_prob_t[:, k] * self.B[k, O[t+1]] * backward_prob[t+1, k] for k in range(self.M)]).transpose()
            double_state_prob.append(double_state_prob_t)

        Pr_O = np.sum(forward_prob[-1])
        double_state_prob = np.array(double_state_prob) / Pr_O
        return double_state_prob

    def evaluate_forward(self, O):
        """
          Given the model (A, B, P) and the observed samples O,
          evaluate the probability that the model generates O using forward probability
        """
        forward_prob = self.get_forward_prob(O)
        return np.sum(forward_prob[-1, :])

    def evaluate_backward(self, O):
        """
          Given the model (A, B, P) and the observed samples O,
          evaluate the probability that the model generates O using backward probability
        """
        backward_prob = self.get_backward_prob(O)
        return np.sum(backward_prob[0, :] * self.B[:, O[0]] * self.P)

    def decoding_Viterbi(self, O):
        """
        Given the model and observed samples O, find the most likely hidden states I that generates O
        I* = arg max_{I} Pr(I|O; A, B, P)

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

    def learning_EM(self, O_array):
        """
        learn the parameter of the model using Baum-Welch (i.e., EM) algorithm
        :param O_array: a list of observed sequences: num_seqs X T
        :return A, B, P: (A, B, P) = arg max Pr(O|A, B, P)
        """
        D = len(O_array)
        pass

    def learning_supervised(self, I_array, O_array):
        """
        Leraning the parameters of the model given the complete data
        :param I_array:
        :param O_array:
        :return:
        """
        assert np.shape(I_array) == np.shape(O_array)

        # estimate P
        P = np.histogram(I_array[:, 0], bins=np.arange(self.M + 0.1), density=True)[0]

        # estimate A and B
        num_seqs, T = np.size(I_array, 0), np.size(I_array, 1)
        num_hidden_samples1 = np.histogram(I_array[:, 0:T-1], bins=np.arange(self.M + 0.1))[0]
        num_hidden_samples2 = num_hidden_samples1 + np.histogram(I_array[:, -1], bins=np.arange(self.M + 0.1))[0]
        A, B = np.zeros((self.M, self.M)), np.zeros((self.M, self.N))
        for i in range(num_seqs):
            for t in range(T-1):
                A[I_array[i, t], I_array[i, t+1]] += 1
                B[I_array[i, t], O_array[i, t]] += 1
            B[I_array[i, -1], O_array[i, -1]] += 1
        A = np.array([A[m, :]/num_hidden_samples1[m] for m in range(self.M)])
        B = np.array([B[m, :]/num_hidden_samples2[m] for m in range(self.M)])
        return A, B, P


def test(hmm_instance, out_samples):
    print('Evaluate_forward:', hmm_instance.evaluate_forward(out_samples))
    print('Evaluate_backward:', hmm_instance.evaluate_backward(out_samples))
    hmm_instance.verify_forward_backward(out_samples)
    print('\nDecoding using Viterbi:', hmm_instance.decoding_Viterbi(out_samples), '==>', out_samples)

    print('\nPosterior probability for a single state')
    single_state_prob = hmm_instance.get_single_state_prob(out_samples)
    print(single_state_prob)
    assert all([is_equal(s, 1.0) for s in np.sum(single_state_prob, axis=1)])

    print('\nPosterior probability for two states')
    double_state_prob = hmm_instance.get_double_state_prob(out_samples)
    for t in range(np.size(double_state_prob, 0)):
        print('t = {0}:'.format(t))
        print(double_state_prob[t, :, :])
        assert is_equal_iterable(single_state_prob[t, :], np.sum(double_state_prob[t, :, :], axis=1))


def test_learning(hmm_instance):
    print('\n=============test learning algorithm=============')
    num_seqs, T = 1000, 1000
    I_list, O_list = [], []
    for i in range(num_seqs):
        hidden_samples, out_samples = hmm_instance.generation(T)
        I_list.append(hidden_samples)
        O_list.append(out_samples)
    A, B, P = hmm_instance.learning_supervised(np.array(I_list), np.array(O_list))
    print(A - hmm_instance.A, '\n', B - hmm_instance.B, '\n', P - hmm_instance.P)


if '__main__' == __name__:
    # 李航-统计学习方法中的盒子与球模型
    A = np.array([(0.5, 0.2, 0.3), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5)])
    B = np.array([(0.5, 0.5), (0.4, 0.6), (0.7, 0.3)])
    P = np.array((0.2, 0.4, 0.4))
    hmm_instance = HMM(A, B, P)

    test_learning(hmm_instance)

    # print('\n=============test 1=============')
    # out_samples = (0, 1, 0)
    # test(hmm_instance, out_samples)

    # print('\n=============test 3=============')
    # A = np.array([[1.0/3] * 3] * 3)
    # B = np.array([[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]])
    # P = np.array([1.0/3] * 3)
    # hmm_instance = HMM(A, B, P)
    #
    # out_samples = [x-1 for x in [1, 1, 1, 1, 2, 1, 2, 2, 2, 2]]
    # # 隐藏状态序列的最大后验估计为[1, 1, 1, 1, 2, 1, 2, 2, 2, 2]
    # test(hmm_instance, out_samples)
