import numpy as np
import sklearn
from numpy import linalg as nplinalg
from scipy import sparse as ssp
from scipy.sparse import linalg
import matplotlib.pyplot as plt

#from sklearn.decomposition import FastICA

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

X_mat = np.zeros((len(user_set), len(movie_set)))
X_flag = np.zeros((len(user_set), len(movie_set)))
Y_mat = ssp.lil_matrix((len(user_set), len(movie_set)))


#X_dense_mat = np.zeros((len(user_set), len(movie_set)))

del f, file_content

f = open("movie_ratings.txt")
file_content = f.readlines()

for row in file_content:
	clean_row = row.rstrip('\n').split(',')
	tmp_row = clean_row[0]
	tmp_col = movie_list.index(clean_row[1])
	X_mat[tmp_row, tmp_col] = clean_row[2]
	X_flag[tmp_row, tmp_col] = 1
	#X_dense_mat[tmp_row, tmp_row] = clean_row[2]
	#print('The value at index: ', tmp_row, tmp_col, 'is: ', clean_row[2], 'Assigned is:', clean_row[2])

#print(movie_list)
action_vec = np.zeros((1, len(movie_list)))
romance_vec = np.zeros((1, len(movie_list)))


f_action = open("action_movies_correct.txt")
action_file_content = f_action.readlines()
for row in action_file_content:
	clean_row = row.rstrip('\n').split(',')
	try:
		action_vec[0, movie_list.index(clean_row[0])] = 1
	except:
		print('Element: ', clean_row[0], 'is not in the list')
	#print(clean_row[0])

f_romance = open("romance_movies_correct.txt")
romance_file_content = f_romance.readlines()
for row in romance_file_content:
	clean_row = row.rstrip('\n').split(',')
	try:
		romance_vec[0, movie_list.index(clean_row[0])] = 1	
	except:
		print('Element: ', clean_row[0], 'is not in the list')

#quit()

for ii in range(0, len(movie_set)):
	tmp_alpha = sum(X_mat[:,ii])/sum(X_flag[:,ii])
	#print('The a_j value for colum j is: ', ii, tmp_alpha)	
	for jj in range(0, len(user_set)):
		if X_flag[jj,ii] == 1:
			#print('Assigned values')
			Y_mat[jj,ii] = X_mat[jj,ii] - tmp_alpha

U, s, V = linalg.svds(Y_mat, k=10, which = 'LM')

print(U.shape, s.shape, V.shape)

dense_U = np.zeros((len(user_set), len(user_set)))
dense_E = np.zeros((len(user_set), len(movie_set)))
dense_V = np.zeros((len(movie_set), len(movie_set)))

dense_U[:,0:10] = U
dense_V[0:10,:] = V
for ii in range(0, len(s)):
	dense_E[ii, ii] = (s[ii])**2

# Find the Feature Vectors from SVD

#P_mat = np.dot(np.dot(np.transpose(dense_V), np.dot(np.transpose(dense_E), dense_E)), dense_V)
P_mat = dense_V

action_cross_corr = np.zeros((1,10))
romance_cross_corr = np.zeros((1,10))
for jj in range(0, 10):
	#tmp_action_vec = np.multiply(np.asarray(action_vec) - np.mean(np.asarray(action_vec)), np.asarray(dense_V[jj,:]) - np.mean(np.asarray(dense_V[jj,:])))
	#tmp_romantic_vec = np.multiply(np.asarray(romance_vec) - np.mean(np.asarray(romance_vec)), np.asarray(dense_V[jj,:]) - np.mean(np.asarray(dense_V[jj,:])))
	#print(tmp_action_vec.shape, tmp_romantic_vec.shape)
	#action_cross_corr[0,jj] = np.mean(np.asarray(np.transpose(tmp_action_vec)))/(np.std(action_vec)*np.std(dense_V[jj,:]))
	#romance_cross_corr[0,jj] = np.mean(np.asarray(np.transpose(tmp_romantic_vec)))/(np.std(romance_vec)*np.std(dense_V[jj,:]))
	action_set = np.multiply(np.asarray(action_vec), np.asarray(dense_V[jj,:]))
	romance_set = np.multiply(np.asarray(romance_vec), np.asarray(dense_V[jj,:]))
	action_set = np.asarray(action_set)
	romance_set = np.asarray(romance_set)
	romance_std = np.std(np.asarray(romance_set))
	action_std = np.std(np.asarray(action_set))
	action_cross_corr[0,jj] = action_std
	romance_cross_corr[0,jj] = romance_std

	#print(A.shape, B.shape)

print(action_cross_corr)
print(romance_cross_corr)	

#action_idx = list(action_cross_corr).index(max(action_cross_corr))
#romance_idx = list(romance_cross_corr).index(max(romance_cross_corr))

#print(action_idx, romance_idx)
#quit()
new_X = np.zeros((2,len(movie_set)))
#new_X[0,:] = dense_V[3,:]
#new_X[1,:] = dense_V[9,:]

#new_X[0,:] = dense_V[6,:]
#new_X[1,:] = dense_V[2,:]

new_X[0,:] = dense_V[0,:]
new_X[1,:] = dense_V[2,:]
new_A = np.dot(np.transpose(new_X), new_X)
L, P = nplinalg.eig(new_A)
final_project = np.dot(P, np.transpose(new_X))

project_x = final_project[:,0]
project_y = final_project[:,1]

plt.scatter(project_x, project_y, alpha = 0.1)
plt.show()
#action_projection = np.dot(P_mat, np.transpose(action_vec))
#romance_projection = np.dot(P_mat, np.transpose(romance_vec))
#print(len(action_projection), len(romance_projection))
#plt.scatter(action_projection, romance_projection)
#plt.show()
#print(P_mat.shape)


