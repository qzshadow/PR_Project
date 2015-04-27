# -*- coding: utf-8 -*-
__author__ = 'Qin'

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
import os

def generate_data(label_encode = True, n_folds = 5, iris_path = './iris.data'):
    label_encode = True
    n_folds = 5

    # Load the Iris flower dataset
    if not os.path.exists(iris_path):
        import urllib
        origin = (
            'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        )
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, iris_path)

    iris = pd.read_csv('./iris.data',
                       names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'],
                       header=None)
    iris = iris.dropna()

    X = np.array(iris[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']])  # features
    y = iris['Species'] # class

    if label_encode:
        # Transform string (nominal) output to numeric
        labels = preprocessing.LabelEncoder().fit_transform(y)
    else:
        labels = y

    # Generate k stratified folds of the data.
    return list(cross_validation.StratifiedKFold(labels, n_folds))