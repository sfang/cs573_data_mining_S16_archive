import csv
import pandas as pd
import numpy as np 
from sklearn import svm
#from scikit-learn import svm
#from scikit import svm

X = np.array([[1,2],
[5,8],
[8,8],
[1,0.6],
[9,11]])
y = [0,1,0,1,0]
clf = svm.SVC(kernel='linear', C = 1.8)
# if you want a linear kernel clf = svm.SVC(kernel='linear', C = ??)
# C = <number> is the relaxation penalty that you must choose (do not run the code with ??)
clf.fit(X,y)

# test1 = np.asarray([0.58,0.76])
# test2 = np.asarray([10.58,10.76])

# test1.reshape(1, -1)
# test2.reshape(1, -1)
print(clf.predict([0.58,0.76]))
print(clf.predict([10.58,10.76]))
# if you are interested in the w coefficients
#w = clf.coef_[0]



