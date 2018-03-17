import numpy as np
import sklearn
#from numpy import linalg as linalg
from scipy import sparse as ssp
from scipy.sparse import linalg
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA


f = open("movie_ratings.txt")
file_content = f.readlines()

total_movie = []
total_user = []

for row in file_content:
	clean_row = row.rstrip('\n').split(',')
	#print(clean_row)
	total_user.append(clean_row[0])
	total_movie.append(clean_row[1])

user_set = set(total_user)
movie_set = set(total_movie)

movie_list = list(movie_set)
#print(len(user_set))
#print(len(movie_set))	

X_mat = ssp.lil_matrix((len(user_set), len(movie_set)))
#X_dense_mat = np.zeros((len(user_set), len(movie_set)))

del f, file_content

f = open("movie_ratings.txt")
file_content = f.readlines()

for row in file_content:
	clean_row = row.rstrip('\n').split(',')
	tmp_row = clean_row[0]
	tmp_col = movie_list.index(clean_row[1])
	X_mat[tmp_row, tmp_col] = clean_row[2]
	#X_dense_mat[tmp_row, tmp_row] = clean_row[2]
	#print('The value at index: ', tmp_row, tmp_col, 'is: ', clean_row[2], 'Assigned is:', clean_row[2])

U, s, V = linalg.svds(X_mat, k=10, which = 'LM')

print(U.shape, s.shape, V.shape)
print(s)

dense_U = np.zeros((len(user_set), len(user_set)))
dense_E = np.zeros((len(user_set), len(movie_set)))
dense_V = np.zeros((len(movie_set), len(movie_set)))

dense_U[:,0:10] = U
dense_V[0:10,:] = V
for ii in range(0, len(s)):
	dense_E[ii, ii] = s[ii]

X_dense_mat = X_mat.toarray()

#X_dense_mat = np.dot(np.dot(dense_U, dense_E), dense_V)

print(X_dense_mat.shape)


#plt.plot(s[0:10])
#plt.ylabel('Standard Deviation For Component j')
#plt.xlabel('The Component j (PCA Component)')
#plt.show()

#quit()
#for ii in s:
#	print(ii)
#
#quit()

ica = FastICA(n_components = 2)
S = ica.fit_transform(X_dense_mat)
#A = ica.get_mixing_matrix()
A = ica.mixing_
print(S.shape)
print(A.shape)

plt.scatter(A[:,0], A[:,1], alpha = 0.1)
plt.scatter(A[movie_list.index('111'),0], A[0,1], color = 'red', s = 100, alpha = 0.7)
plt.scatter(A[movie_list.index('288'),0], A[0,1], color = 'green', s = 100, alpha = 0.7)
plt.scatter(A[movie_list.index('102'),0], A[0,1], color = 'yellow', s = 100, alpha = 0.7)
plt.scatter(A[movie_list.index('291'),0], A[0,1], color = 'black', s = 100, alpha = 0.7)
plt.show()
# # Build the matrix X
# X_mat = np.zeros((1000,40))
# #print(type(X_mat))

# f = open("mystery_data.txt")
# file_content = f.readlines()

# for row in file_content:
# 	#print(row.rstrip('\n').split(','))
# 	clean_row = row.rstrip('\n').split(',')
# 	#print(clean_row[0])
# 	tmp_row = int(clean_row[0])
# 	tmp_col = int(clean_row[1])
# 	X_mat[tmp_row,tmp_col] = float(clean_row[2])
# 	#print('The value at index: ', tmp_row, tmp_col, 'is: ', clean_row[2], 'Assgned is:', clean_row[2])

# # Using the regular SVD from Numpy Module
# U, s, V = np.linalg.svd(X_mat, full_matrices = True)
# print(U.shape, V.shape, s.shape)
# print(s)

# A = np.dot(X_mat, np.transpose(X_mat))
# # Find the eigenvector matrices and eigen values
# L, P = np.linalg.eig(A)
# #print(L)
# #print(P[:,0])

# tmp_interm = np.dot(P[:,0], A)
# tmp_interm = np.dot(tmp_interm, np.transpose(P[:,0]))
# print(tmp_interm)

# std_componet_vec = []
# for idx in range(0, len(L)):
# 	tmp_interm = np.dot(P[:,idx], A)
# 	tmp_interm = np.dot(tmp_interm, np.transpose(P[:,idx]))
# 	tmp_std = (tmp_interm)**(0.5)
# 	std_componet_vec.append(tmp_std)

# print(len(std_componet_vec))
# plt.plot(std_componet_vec[0:40])
# plt.ylabel('Standard Deviation For Component j')
# plt.xlabel('The Component j (PCA Component)')
# plt.show()
