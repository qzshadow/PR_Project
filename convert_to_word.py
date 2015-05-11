# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:10:31 2015

@author: Qin
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def convert_np_array_to_word_equation(array,pre='%.2f'):
    result = '■('
    for row in array:
        for j,col in enumerate(row):
            result += (pre % col)
            if not j == len(row)-1:
                result += '&'
        if not np.array_equal(row,array[-1]):
            result += '@'
    result+=')'
    return result
    
def reg(array):
    result = np.asarray(array,dtype='f64')
    row = len(array)
    col = len(array[0])
    for i in range(row):
        for j in range(col):
            result[i,j] = array[i,j]/(array[i,i]+array[j,j]-array[i,j])
    return result

transform = TfidfTransformer()
vectorizer = CountVectorizer(min_df=1)
corpus = ['D D A B','C A A D','D C B B','A B C']
X = vectorizer.fit_transform(corpus)
X.toarray()
#print(convert_np_array_to_word_equation(X.toarray()))


