# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:29:54 2015

@author: Qin
"""

#pca version of iris dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib import animation

from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
print(X_iris.shape, y_iris.shape)
print(X_iris[0], y_iris[0])
n_components = 3
estimator = PCA(n_components=n_components)
X_pca = estimator.fit_transform(X_iris)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_pca_scatter():
    colors = ['red', 'greenyellow', 'blue']
    for i in np.arange(len(colors)):
        px = X_pca[:, 0][y_iris == i]
        py = X_pca[:, 1][y_iris == i]
        pz = X_pca[:, 2][y_iris == i]
        ax.scatter(px, py, pz, c=colors[i])
    ax.legend(iris.target_names)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    plt.show()
    
def animate(i):
    ax.view_init(elev=10., azim=i)
    

anim = animation.FuncAnimation(fig, animate, init_func=plot_pca_scatter,
                               frames=360, interval = 20, blit =True)
#save
anim.save('plot_iris.mp4', fps = 30, extra_args=['-vcodec', 'libx264'])
                        

