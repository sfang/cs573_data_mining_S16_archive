import numpy as np
import sklearn
from numpy import linalg as linalg
from scipy import sparse as ssp
from scipy.sparse import linalg
import matplotlib.pyplot as plt


# Build the matrix X
X_mat = np.zeros((1000,40))
#print(type(X_mat))

f = open("mystery_data.txt")
file_content = f.readlines()

for row in file_content:
	#print(row.rstrip('\n').split(','))
	clean_row = row.rstrip('\n').split(',')
	#print(clean_row[0])
	tmp_row = int(clean_row[0])
	tmp_col = int(clean_row[1])
	X_mat[tmp_row,tmp_col] = float(clean_row[2])
	#print('The value at index: ', tmp_row, tmp_col, 'is: ', clean_row[2], 'Assgned is:', clean_row[2])

# Using the regular SVD from Numpy Module
U, s, V = np.linalg.svd(X_mat, full_matrices = True)
print(U.shape, V.shape, s.shape)
print(s)

A = np.dot(X_mat, np.transpose(X_mat))
# Find the eigenvector matrices and eigen values
L, P = np.linalg.eig(A)
#print(L)
#print(P[:,0])

tmp_interm = np.dot(P[:,0], A)
tmp_interm = np.dot(tmp_interm, np.transpose(P[:,0]))
print(tmp_interm)

std_componet_vec = []
for idx in range(0, len(L)):
	tmp_interm = np.dot(P[:,idx], A)
	tmp_interm = np.dot(tmp_interm, np.transpose(P[:,idx]))
	tmp_std = (1/(len(s))*tmp_interm)**(0.5)
	std_componet_vec.append(tmp_std)

print(len(std_componet_vec))
plt.plot(std_componet_vec[0:40])
plt.ylabel('Standard Deviation For Component j')
plt.xlabel('The Component j (PCA Component)')
plt.show()
