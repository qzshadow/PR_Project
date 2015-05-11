# -*- coding: utf-8 -*-
__author__ = 'Qin'

from sklearn import metrics
import numpy as np
import random
from import_iris_data import generate_data
from ANN_model import MLP_1HL
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

# Generate classification objects.

X,labels,n_folds,skf = generate_data(label_encode=True,n_folds=10,iris_path='./iris.data')
# Generate arrays for meta-level training and testing sets, which are n x len(clfs).
scores_nn = np.zeros(n_folds) # scores for nn
result = np.zeros((7,7))
for hid_size_idx, hidden_layer_size in enumerate(np.linspace(10,130,7)):
    for reg_idx, reg_lambda in enumerate(np.hstack((np.linspace(0.01,0.1,3,endpoint=False),
                                                    np.linspace(0.1,1,4)))):
        train_result = np.zeros(5)
        for train_idx in np.arange(5):
            nn = MLP_1HL(reg_lambda=reg_lambda,epsilon_init=0.16,hidden_layer_size=int(hidden_layer_size),opti_method='TNC',maxiter=500,load_theta0=False)
            print('Training classifiers...')
            # Iterate over the folds, each with training set and validation set indicies.
            for i, (train_index, test_index) in enumerate(skf):
                print('  Fold {0}'.format(i))
            
                # Generate the training set for the fold.
                X_train = X[train_index]
                y_train = labels[train_index]
            
                # Generate the testing set for the fold.
                X_test = X[test_index]
                y_test = labels[test_index]
            
                # Train the models on the training set.
                # We time the training using the built-in timeit magic function.
                print('    Neural Network: ',
                nn.fit(X_train, y_train))
            
                # Evaluate the models on the testing set.
                scores_nn[i] = metrics.accuracy_score(y_test, nn.predict(X_test))
            print('Done training classifiers.')
            print("\n")
            
            # The mean of the scores on the testing set.
            print('Artificial Neural Network Accuracy = %s' % (scores_nn.mean(axis=0)))
            train_result[train_idx] = scores_nn.mean(axis=0)
        result[hid_size_idx][reg_idx] = train_result.mean(axis=0)

X = np.linspace(10,130,7)
Y = np.hstack((np.linspace(0.01,0.1,3,endpoint=False),np.linspace(0.1,1,4)))
X, Y = np.meshgrid(X,Y)
fig, ax = plt.subplots()
p = ax.pcolor(X,Y,result,cmap=cm.RdBu)
fig.colorbar(p,ax=ax)
plt.xlabel('hidden layer size')
plt.ylabel('regularization factor')
plt.title('accuracy plot by changing maxiter and hidden_layer_size')
plt.legend()
plt.show()

def myShuffle(X,y):
    num_features = len(X[0])
    sample = np.hstack((X,y))
    random.shuffle(sample)
    return sample[:,:num_features],sample[:,num_features:]