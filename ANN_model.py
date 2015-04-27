# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:13:51 2015
Functions tp implement an artificial neural network(multi-layer perceptron)
@author: qinzhou
"""

import numpy as np
from scipy import optimize


class MLP_1HL():
    """Implements of an one hidden-layer MLP
    Reference:
    http://www.deeplearning.net/tutorial/mlp.html#mlp
    http://nbviewer.ipython.org/github/cse40647/cse40647/blob/sp.14/33%20-%20Artificial%20Neural%20Networks.ipynb
    """

    def __init__(self, max_iter=500, reg_lambda=0.01,
                 hidden_layer_size=25, opt_method = "TNC"):
        self.reg_lambda = reg_lambda
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter
        self.active_fun = self.sigmoid
        self.active_fun_prime = self.sigmoid_primer
        self.opt_method = opt_method

    def sigmoid(self, z):
        """
        implement of logistic function
        :param z: output of last_layer
        :return: activation of last_layer
        """
        return 1 / (1 + np.exp(-z))

    def tanh_fun(self, z):
        a = np.exp(z)
        b = np.exp(-z)
        return (a - b) / ( a + b)

    def tanh_primer(self, z):
        return 1 - self.tanh_fun(z) ** 2

    def sigmoid_primer(self, z):
        """
        implement of derivation of logistic function
        :param z:
        :return:
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def weight_init(self, rng, n_in, n_out):
        return np.asarray(
            rng.uniform(
                low=-np.sqrt(6 / (n_in + n_out)),
                high=np.sqrt(6 / (n_in + n_out)),
                size=(n_out, n_in+1)
            )
        )

    def pack_thetas(self, theta_in_hidden, theta_hidden_out):
        return np.concatenate((theta_in_hidden.reshape(-1), theta_hidden_out.reshape(-1)))

    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, output_layer_size):
        theta_in_hid_start = 0
        theta_in_hid_end = hidden_layer_size * (input_layer_size + 1)
        theta_in_hid = thetas[theta_in_hid_start:theta_in_hid_end + 1].reshape(input_layer_size, hidden_layer_size + 1)
        theta_hid_out = thetas[theta_in_hid_end:].reshape(hidden_layer_size + 1, output_layer_size)
        return theta_in_hid, theta_hid_out

    def MLP_forward(self, X, t_1, t_2):
        """

        :param X: input n*m
        :param t_1: hid_size * (input_size+1) h*(m+1)
        :param t_2: out_size * (hid_size +1) o*(h+1)
        :return:
        a1: transformed input(add w_0) n*(m+1)
        z2: output of hidden layer n*h
        a2: transformed output of hidden layer by activation function n*(h+1)
        z3: output of output layer n*o
        a3: transformed output of output layer by activation function n*o
        """
        if len(X.shape) == 1:
            ones = np.array([1])
        else:
            n, m = X.shape
            ones = np.ones(n).reshape((n, 1))
        # input layer
        a1 = np.hstack((ones, X))

        # hidden layer
        z2 = np.dot(a1, t_1.T)
        a2 = np.hstack((ones, self.active_fun(z2)))

        # output layer
        z3 = np.dot(a2, t_2.T)
        a3 = self.active_fun(z3)

        return a1, z2, a2, z3, a3

    def MLP_Cost(self, input_layer_size, hidden_layer_size, output_layer_size, X, y, thetas, reg_lambda):
        """

        :param input_layer_size: input layer size
        :param hidden_layer_size: hidden layer size
        :param output_layer_size: output layer size(number of labels)
        :param X: input samples n*m
        :param y: the label of these samples n*1
        :param thetas: weights in the network (input_layer_size+1)*hidden_layer_size+(hidden_layer_size+1)*output_layer_size
        :param reg_lambda: regulazation parameters
        :return:
        n: number of samples
        Y: reshape of labels, for example if y = [1,2,3,4] then Y = [[1000],[0100],[0010],[0001]] n*o

        """
        t_1, t_2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, output_layer_size)
        n = X.shape[0]
        Y = np.eye(output_layer_size)[y]

        # perform forward propogation
        a1, z2, a2, z3, a3 = self.MLP_forward(X, t_1, t_2)

        # calculate the cost
        costPos = - Y * np.log2(a3).T
        costNeg = - (1 - Y) * np.log2(1 - a3).T
        cost = costPos + costNeg
        J = np.sum(cost) / n

        if reg_lambda:
            t_1p = t_1[:, 1:].reshape(-1)
            t_2p = t_2[:, 1:].reshape(-1)
            punish_lambda = np.dot(t_1p, t_1p.T) + np.dot(t_2p, t_2p.T)
            J += reg_lambda * punish_lambda

        return J

    def MLP_BP(self, input_layer_size, hidden_layer_size, output_layer_size, X, y, thetas, reg_lambda):
        t_1, t_2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, output_layer_size)
        n = X.shape[0]
        Y = np.eye(output_layer_size)[y]

        # perform back propagation
        t1 = t_1[:, 1:]
        t2 = t_2[:, 1:]
        Delta1, Delta2 = 0, 0
        for i, x in enumerate(X):
            a1, z2, a2, z3, a3 = self.MLP_forward(x, t1, t2)
            d3 = (a3 - Y[i, :]) * self.active_fun_prime(a3)  # delta of output layer 1*o
            d2 = np.dot(d3.T, t_2) * self.active_fun_prime(z2)  # delta of hidden layer

            # weight update
            Delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])
            Delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])

        # the unregularized gradients of each weight
        Theta1_grad = Delta1 * (1 / n)
        Theta2_grad = Delta2 * (1 / n)

        if reg_lambda:
            Theta1_grad = Theta1_grad[:, 1:] + reg_lambda * t_1
            Theta2_grad = Theta2_grad[:, 1:] + reg_lambda * t_2

        return self.pack_thetas(Theta1_grad, Theta2_grad)

    def fit(self, X, y):
        """
        fit the modle to samples
        :param X: the training samples n*m
        :param y: the label of training samples n*1
        :return:
        """
        num_samples = X.shape[0]
        num_features = X.shape[1]
        num_labels = len(set(y))

        # ramdom initialize the weights
        Theta1_0 = self.weight_init(np.random.randint(1,100),num_features,self.hidden_layer_size)
        Theta2_0 = self.weight_init(np.random.randint(1,100),self.hidden_layer_size,num_labels)

        Theta_pack = self.pack_thetas(Theta1_0,Theta2_0)

        # Minimize the objective (cost) function and return the resulting thetas.
        options = {'max_iter': self.max_iter}
        _res = optimize.minimize(self.MLP_Cost, Theta_pack, jac=self.MLP_BP, method=self.opt_method,
                                 args=(num_features, self.hidden_layer_size, num_labels, X, y, 0), options=options)

        # set the fitted thetas
        self.t1, self.t2 = self.unpack_thetas(_res,num_features,self.hidden_layer_size,num_labels)

    def predict_proba(self, X):
        _,_,_,_,h = self.MLP_forward(X,self.t1, self.t2)
        return h

    def predict(self, X):
        return self.predict_proba(X).argmax(0)






















        



