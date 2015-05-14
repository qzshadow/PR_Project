import numpy as np
from scipy import optimize
from sklearn.base import ClassifierMixin
import os


class MLP_1HL(ClassifierMixin):
    """Implements an artifical neural network (ANN) with one hidden layer.


    """

    def __init__(self, reg_lambda=0, epsilon_init=0.12, hidden_layer_size=25,
                 opti_method='TNC', maxiter=500, load_theta0 = False):
        self.reg_lambda = reg_lambda  # weight for the logistic regression cost
        self.epsilon_init = epsilon_init  # learning rate
        self.hidden_layer_size = hidden_layer_size  # size of the hidden layer
        self.activation_func = self.sigmoid  # activation function
        self.activation_func_prime = self.sigmoid_prime  # derivative of the activation function
        self.method = opti_method  # optimization method
        self.maxiter = maxiter  # maximum number of iterations
        self.load_theta0 = load_theta0 #for test only

    def sigmoid(self, z):
        """Returns the logistic function."""
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Returns the derivative of the logistic function."""
        sig = self.sigmoid(z)

        return sig * (1 - sig)


    def sumsqr(self, a):
        """Returns the sum of squared values."""
        return np.sum(a ** 2)

    def rand_init(self, l_in, l_out):
        """Generates an (l_out x l_in+1) array of thetas (threshold values), 
        each initialized to a random number between minus epsilon and epsilon.

        Note that there is one theta matrix per layer. The size of each theta 
        matrix depends on the number of activation units in its corresponding 
        layer, so each matrix may be of a different size.

        Returns
        -------
        Randomly initialized thetas (threshold values).
        """
        return np.random.rand(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init

    # Pack thetas (threshold values) into a one-dimensional array.
    def pack_thetas(self, t1, t2):
        """Packs (unrolls) thetas (threshold values) from matrices into a 
        single vector.

        Note that there is one theta matrix per layer. To use an optimization 
        technique that minimizes the error, we need to pack (unroll) the 
        matrices into a single vector.

        Parameters
        ----------
        t1 : array
            Unpacked (rolled) thetas (threshold values) between the input 
            layer and hidden layer.

        t2 : array
            Unpacked (rolled) thetas (threshold values) between the hidden 
            layer and output layer.

        Returns
        -------
        Packed (unrolled) thetas (threshold values).
        """
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))

    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        """Unpacks (rolls) thetas (threshold values) from a single vector into 
        a multi-dimensional array (matrices).

        Parameters
        ----------
        thetas : array
            Packed (unrolled) thetas (threshold values).

        input_layer_size : integer
            Number of nodes in the input layer.

        hidden_layer_size : integer
            Number of nodes in the hidden layer.

        num_labels : integer
            Number of classes.

        Returns
        -------
        t1 : array
            Unpacked (rolled) thetas (threshold values) between the input 
            layer and hidden layer.

        t2 : array
            Unpacked (rolled) thetas (threshold values) between the hidden 
            layer and output layer.
        """
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))

        return t1, t2

    def _forward(self, X, t1, t2):
        """Forward propogation.

        Parameters
        ----------
        X : array, shape=(n, m)
            The feature data for which to compute the predicted output 
            probabilities.

        t1 : array
            Unpacked (rolled) thetas (threshold values) between the input 
            layer and hidden layer.

        t2 : array
            Unpacked (rolled) thetas (threshold values) between the hidden 
            layer and output layer.

        Returns
        -------
        a1 : array
            The output activation of units in the input layer.

        z2 : array
            The input activation of units in the hidden layer.

        a2 : array
            The output activation of units in the hidden layer.

        z3 : array
            The input activation of units in the output layer.

        a3 : array
            The output activation of units in the output layer.
        """
        m = X.shape[0]
        ones = None
        if len(X.shape) == 1:
            ones = np.array(1).reshape(1, )
        else:
            ones = np.ones(m).reshape(m, 1)

        # Input layer.
        a1 = np.hstack((ones, X))

        # Hidden layer.
        z2 = np.dot(t1, a1.T)
        a2 = self.activation_func(z2)
        a2 = np.hstack((ones, a2.T))

        # Output layer.
        z3 = np.dot(t2, a2.T)
        a3 = self.activation_func(z3)

        return a1, z2, a2, z3, a3

    def cost(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        """Returns the total cost using a generalization of the regularized 
        logistic regression cost function.
        """
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        Y = np.eye(num_labels)[y]

        _, _, _, _, h = self._forward(X, t1, t2)
        costPositive = -Y * np.log(h).T  # the cost when y is 1
        costNegative = (1 - Y) * np.log(1 - h).T  # the cost when y is 0
        cost = costPositive - costNegative  # the total cost
        J = np.sum(cost) / m  # the (unregularized) cost function

        # For regularization.
        if reg_lambda != 0:
            t1f = t1[:, 1:]
            t2f = t2[:, 1:]
            reg = (self.reg_lambda / (2 * m)) * (self.sumsqr(t1f) + self.sumsqr(t2f))  # regularization term
            J = J + reg  # the regularized cost function

        return J

    def back_propagation(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        """Returns the Jacobian matrix (the matrix of all first-order partial 
        derivatives) of the cost function.
        """
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        t1f = t1[:, 1:]  # threshold values between the input layer and hidden layer (excluding the bias input)
        t2f = t2[:, 1:]  # threshold values between the hidden layer and output layer (excluding the bias input)
        Y = np.eye(num_labels)[y]

        Delta1, Delta2 = 0, 0  # initialize matrix Deltas (cost function gradients)
        # Iterate over the instances.
        for i, row in enumerate(X):
            # Forwardprop.
            a1, z2, a2, z3, a3 = self._forward(row, t1, t2)

            # Backprop.
            d3 = a3 - Y[i, :].T  # activation error (delta) in the output layer nodes
            d2 = np.dot(t2f.T, d3) * self.activation_func_prime(
                z2)  # activation error (delta) in the hidden layer nodes

            # Update matrix Deltas (cost function gradients).
            Delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])
            Delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])

        # The (unregularized) gradients for each theta.
        Theta1_grad = (1 / m) * Delta1
        Theta2_grad = (1 / m) * Delta2

        # For regularization.
        if reg_lambda != 0:
            # The regularized gradients for each theta (excluding the bias input).
            Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (reg_lambda / m) * t1f
            Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (reg_lambda / m) * t2f

        return self.pack_thetas(Theta1_grad, Theta2_grad)

    def fit(self, X, y):
        """Fit the model given predictor(s) X and target y.

        Parameters
        ----------
        X : array, shape=(n, m)
            The feature data for which to compute the predicted output.

        y : array, shape=(n,1)
            The actual outputs (class data).
        """
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))

        # Initialize (random) thetas (threshold values).
        if self.load_theta0 and os.path.exists('./thetas0.npy'):
            thetas0 = np.fromfile('./thetas0.npy')
        elif self.load_theta0 and not os.path.exists('./theta0.npy'):
            print('warning!!! load thetas from file error! using random thetas')
            theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
            theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
            thetas0 = self.pack_thetas(theta1_0, theta2_0)
        else:
            theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
            theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
            thetas0 = self.pack_thetas(theta1_0, theta2_0)

        # Minimize the objective (cost) function and return the resulting thetas.
        options = {'maxiter': self.maxiter}
        _res = optimize.minimize(self.cost, thetas0, jac=self.back_propagation, method=self.method,
                                 args=(input_layer_size, self.hidden_layer_size, num_labels, X, y, self.reg_lambda), options=options)

        # Set the fitted thetas.
        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)

    def predict(self, X):
        """Predict labels with the fitted model on predictor(s) X.

        Parameters
        ----------
        X : array, shape=(n, m)
            The feature data for which to compute the predicted outputs.

        Returns
        -------
        The predicted labels for each instance X.
        """
        return self.predict_proba(X).argmax(0)

    def predict_proba(self, X):
        """Predict label probabilities with the fitted model on predictor(s) X.

        The probabilities are computed as the output activation of units in 
        the output layer.

        Parameters
        ----------
        X : array, shape=(n, m)
            The feature data for which to compute the predicted output 
            probabilities.

        Returns
        -------
        h : array, shape=(n, 1)
            The predicted label probabilities for each instance X.
        """
        _, _, _, _, h = self._forward(X, self.t1, self.t2)

        return h