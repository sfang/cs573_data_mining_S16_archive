import csv
import pandas as pd
import numpy as np 
#from sklearn import svm
import networkx as nx
#from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from numpy.linalg import inv
from scipy import linalg
from numpy import *
import math
# with open('Bank_Data_Train.csv','rb') as training:
# 	training_reader = csv.reader(training)
# 	#for train_row in training_reader:
# 		#print train_row

# with open('Bank_Data_Test.csv','rb') as testing:
# 	testing_reader = csv.reader(testing)
# 	#for test_row in testing_reader:
# 		#print test_row		

# Preprocessing the data

# Load 1-of-K using Pandas:
def data_processing(filename):
    df = pd.read_csv(filename)
   # remember: always use an intercept for SVM and Logistic regression
    df['intercept'] = 1.0
    return (df)

  
train = data_processing('Bank_Data_Train.csv')
test = data_processing('Bank_Data_Test.csv')

# Encoding Loan Purpose 1-K vector
encoded_loan = pd.get_dummies(pd.concat([train['Loan Purpose'],test['Loan Purpose']], axis=0),\
prefix='Loan Purpose', dummy_na=True)
train_rows = train.shape[0]
train_encoded_loan = encoded_loan.iloc[:train_rows, :]
test_encoded_loan = encoded_loan.iloc[train_rows:, :]
loan_shape = train_encoded_loan.shape
loan_1K = loan_shape[1]
print 'The loan purpose length of 1-K is: ', loan_1K
loan_entries = []
for row in train_encoded_loan:
	loan_entries.append(row)

# Encoding FICO 1-K vector
encoded_FICO = pd.get_dummies(pd.concat([train['FICO Range'],test['FICO Range']], axis=0),\
prefix='FICO Range', dummy_na=True)
train_rows = train.shape[0]
train_encoded_FICO = encoded_FICO.iloc[:train_rows, :]
test_encoded_FICO = encoded_FICO.iloc[train_rows:, :]

FICO_shape = train_encoded_FICO.shape
FICO_1K = FICO_shape[1]
# print test_encoded_FICO.shape
# print type(train_encoded_FICO)
print 'The FICO range length of 1-K is: ', FICO_1K
# cnt = 0
FICO_entries = []
for row in train_encoded_FICO:
 	#print row
 	FICO_entries.append(row)
# 	cnt += 1
#print cnt
# print FICO_entries
# k_vec =  train_encoded_FICO['FICO Range_660-664']
# print k_vec[1]
# #for element in k_vec:
# 	#print element

# Preprocessing the rest of parameters

amount_request = []
interest_rate = []
loan_length = []
month_pay = []
total_fund = []
DIT_ratio = []
decision = []

row_cnt = 0
with open('Bank_Data_Train.csv','rb') as training:
	training_reader = csv.reader(training) 	
 	for train_row in training_reader:
 		if row_cnt > 0:
 			amount_request.append(float(train_row[1]))
 			interest_rate.append(float(train_row[2]))
 			loan_length.append(float(train_row[3]))
 			month_pay.append(float(train_row[6]))
 			total_fund.append(float(train_row[7]))
 			DIT_ratio.append(float(train_row[8]))
 			decision.append(float(train_row[10]))
 		row_cnt += 1		

# print amount_request[0]
# print interest_rate[0]
# print loan_length[0]
# print month_pay[0]
# print total_fund[0]
# print DIT_ratio[0]
# print decision[0]
# print len(amount_request)

# for ii in range(0, len(amount_request)):
# 	if amount_request[ii] != total_fund[ii]:
# 		print 'Not equal'

# with open('Bank_Data_Test.csv','rb') as testing:
# 	testing_reader = csv.reader(testing)
# 	#for test_row in testing_reader:
# 		#print test_row	

# num_bins = 10
# values = np.asarray(amount_request)
# values = (values - min(values)) / (max(values) - min(values))
# bins = np.linspace(0, 1, 11)
# [freq, bins] = np.histogram(values, bins)
# print freq
# print sum(freq)
# print bins

# Phi = np.zeros((len(amount_request), (loan_1K + FICO_1K + 6)))
# phi_shape = Phi.shape

# prob_yes = sum(decision)/len(decision)
# prob_no = 1 - sum(decision) / len(decision)

# Phi[:,0] = values
	
# print Phi[:,0]

# print prob_yes
# print prob_no

# k_vec1 = train_encoded_FICO['FICO Range_660-664']
# k_vec2 = train_encoded_FICO['FICO Range_825-829']

# # X = np.array(train_encoded_FICO['FICO Range_660-664'])
Y = np.array(decision)
# X = np.array([k_vec1, k_vec2])
#Y = np.array([1, 2, 3, 4, 4, 5])
# X.append(k_vec1)
# X_tran = X.transpose()
# print Y.shape
# print X_tran.shape

# clf = BernoulliNB()
# clf.fit(X_tran, Y)

X_matrix = np.zeros((len(amount_request), (FICO_1K + loan_1K + 60)))
#tmp_name = FICO_entries

# Encode FICO into feature matrix
X_matrix[:,0:FICO_1K] = train_encoded_FICO
# Encode Loan purpose into feature matrix
X_matrix[:,FICO_1K:(loan_1K + FICO_1K)] = train_encoded_loan

range_amount_request = max(amount_request) - min(amount_request)
range_interest_rate = max(interest_rate) - min(interest_rate)
range_loan_length = max(loan_length) - min(loan_length)
range_month_pay = max(month_pay) - min(month_pay)
range_total_fund = max(total_fund) - min(total_fund) 
range_DIT_ratio = max(DIT_ratio) - min(DIT_ratio) 



# clf = BernoulliNB() # Create a Binary NBC classifier
# clf.fit(X_matrix, Y)

# predict_results = (clf.predict(X_matrix))
# print sum(abs(Y - predict_results))/len(Y)

# Encoding Amount Requested
for ii in range(0, len(amount_request)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_amount = (amount_request[ii] - min(amount_request)) / range_amount_request
	while tmp_bin < 0.9:
		if tmp_amount < tmp_bin + 0.1:
			X_matrix[ii, (loan_1K + FICO_1K + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break
		if tmp_amount > 1:
			X_matrix[ii, (loan_1K + FICO_1K + 9)] = 1
			print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1	

# Encoding Interest Rate
for ii in range(0, len(interest_rate)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_interest_rate = (interest_rate[ii] - min(interest_rate)) / range_interest_rate
	while tmp_bin < 0.9:
		if tmp_interest_rate < tmp_bin + 0.1:
			X_matrix[ii, (loan_1K + FICO_1K + 10 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 10)
			break
		if tmp_interest_rate > 1:
			X_matrix[ii, (loan_1K + FICO_1K + 10 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1	

# print loan_1K + FICO_1K + 10 + 9
# print loan_1K + FICO_1K

# Encoding Loan Length
for ii in range(0, len(loan_length)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_loan_length = (loan_length[ii] - min(loan_length)) / range_loan_length
	while tmp_bin < 0.9:
		if tmp_loan_length < tmp_bin + 0.1:
			X_matrix[ii, (loan_1K + FICO_1K + 20 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 20)
			break
		if tmp_loan_length > 1:
			X_matrix[ii, (loan_1K + FICO_1K + 20 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1	

# Encoding Loan Length
for ii in range(0, len(month_pay)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_month_pay = (month_pay[ii] - min(month_pay)) / range_month_pay
	while tmp_bin < 0.9:
		if tmp_month_pay < tmp_bin + 0.1:
			X_matrix[ii, (loan_1K + FICO_1K + 30 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 30)
			break
		if tmp_month_pay > 1:
			X_matrix[ii, (loan_1K + FICO_1K + 30 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1		

# Encoding Total Fund
for ii in range(0, len(total_fund)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_total_fund = (total_fund[ii] - min(total_fund)) / range_total_fund
	while tmp_bin < 0.9:
		if tmp_total_fund < tmp_bin + 0.1:
			X_matrix[ii, (loan_1K + FICO_1K + 40 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 40)
			break
		if tmp_total_fund > 1:
			X_matrix[ii, (loan_1K + FICO_1K + 40 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1		

# Encoding DIT 
for ii in range(0, len(DIT_ratio)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_DIT_ratio = (DIT_ratio[ii] - min(DIT_ratio)) / range_DIT_ratio
	#print tmp_DIT_ratio
	while tmp_bin < 0.9:
		if tmp_DIT_ratio < tmp_bin + 0.1:
			X_matrix[ii, (loan_1K + FICO_1K + 50 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 50)
			break
		if tmp_DIT_ratio > 1:
			X_matrix[ii, (loan_1K + FICO_1K + 50 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1					

#print (FICO_1K+loan_1K+59)
#for ii in range(0,111):
#	print sum(X_matrix[:,ii]), len(X_matrix[:,ii])		

## Configure NBC binary classifier
NBC_clf = BernoulliNB() # Create a Binary NBC classifier
NBC_clf.fit(X_matrix, Y)

# print X_matrix.shape
# print Y.shape

# predict_results = (clf.predict(X_matrix))
# print sum(abs(Y - predict_results))/len(Y)


#from scikit-learn import svm
#from scikit import svm

# for ii in range(20,30):
# 	c_value = ii * 0.1 
# 	svm_clf = svm.SVC(kernel='rbf', C = c_value)
# # if you want a linear kernel clf = svm.SVC(kernel='linear', C = ??)
# # C = <number> is the relaxation penalty that you must choose (do not run the code with ??)
# 	svm_clf.fit(X_matrix,decision)

# # test1 = np.asarray([0.58,0.76])
# # test2 = np.asarray([10.58,10.76])

# # test1.reshape(1, -1)
# # test2.reshape(1, -1)
# 	predict_results = (svm_clf.predict(X_matrix))
# 	print 'The prediction accuracy is:', sum(abs(Y - predict_results))/len(Y)

test_amount_request = []
test_interest_rate = []
test_loan_length = []
test_month_pay = []
test_total_fund = []
test_DIT_ratio = []

row_cnt = 0
with open('Bank_Data_Test.csv','rb') as testing:
	testing_reader = csv.reader(testing) 	
 	for test_row in testing_reader:
 		if row_cnt > 0:
 			test_amount_request.append(float(test_row[1]))
 			test_interest_rate.append(float(test_row[2]))
 			test_loan_length.append(float(test_row[3]))
 			test_month_pay.append(float(test_row[6]))
 			test_total_fund.append(float(test_row[7]))
 			test_DIT_ratio.append(float(test_row[8]))
 		row_cnt += 1	

# Build the Feature Matrix for Testing Data
X_matrix_test = np.zeros((len(test_amount_request), (FICO_1K + loan_1K + 60)))
# Encode FICO into feature matrix
X_matrix_test[:,0:FICO_1K] = test_encoded_FICO
# Encode Loan purpose into feature matrix
X_matrix_test[:,FICO_1K:(loan_1K + FICO_1K)] = test_encoded_loan

# Encoding Amount Requested
for ii in range(0, len(test_amount_request)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_amount = (test_amount_request[ii] - min(amount_request)) / range_amount_request
	while tmp_bin < 0.9:
		if tmp_amount < tmp_bin + 0.1:
			X_matrix_test[ii, (loan_1K + FICO_1K + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break
		if tmp_amount > 1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 9)] = 1
			print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1	

# Encoding Interest Rate
for ii in range(0, len(test_interest_rate)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_interest_rate = (test_interest_rate[ii] - min(interest_rate)) / range_interest_rate
	while tmp_bin < 0.9:
		if tmp_interest_rate < tmp_bin + 0.1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 10 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 10)
			break
		if tmp_interest_rate > 1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 10 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1	

# print loan_1K + FICO_1K + 10 + 9
# print loan_1K + FICO_1K

# Encoding Loan Length
for ii in range(0, len(test_loan_length)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_loan_length = (test_loan_length[ii] - min(loan_length)) / range_loan_length
	while tmp_bin < 0.9:
		if tmp_loan_length < tmp_bin + 0.1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 20 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 20)
			break
		if tmp_loan_length > 1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 20 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1	

# Encoding Loan Length
for ii in range(0, len(test_month_pay)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_month_pay = (test_month_pay[ii] - min(month_pay)) / range_month_pay
	while tmp_bin < 0.9:
		if tmp_month_pay < tmp_bin + 0.1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 30 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 30)
			break
		if tmp_month_pay > 1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 30 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1		

# Encoding Total Fund
for ii in range(0, len(test_total_fund)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_total_fund = (test_total_fund[ii] - min(total_fund)) / range_total_fund
	while tmp_bin < 0.9:
		if tmp_total_fund < tmp_bin + 0.1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 40 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 40)
			break
		if tmp_total_fund > 1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 40 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1		

# Encoding DIT 
for ii in range(0, len(test_DIT_ratio)):
	tmp_bin = 0
	tmp_idx = 0
	tmp_DIT_ratio = (test_DIT_ratio[ii] - min(DIT_ratio)) / range_DIT_ratio
	#print tmp_DIT_ratio
	while tmp_bin < 0.9:
		if tmp_DIT_ratio < tmp_bin + 0.1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 50 + tmp_idx)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx + 50)
			break
		if tmp_DIT_ratio > 1:
			X_matrix_test[ii, (loan_1K + FICO_1K + 50 + 9)] = 1
			#print 'Index: ', ii, 'Assigned: ', (loan_1K + FICO_1K + tmp_idx)
			break	
		tmp_idx += 1
		tmp_bin += 0.1

# for ii in range(0,111):
# 	print sum(X_matrix_test[:,ii]), len(X_matrix_test[:,ii])	


# svm_clf = svm.SVC(kernel='rbf', C = 1000.0)
# # if you want a linear kernel clf = svm.SVC(kernel='linear', C = ??)
# # C = <number> is the relaxation penalty that you must choose (do not run the code with ??)
# svm_clf.fit(X_matrix,decision)

# test1 = np.asarray([0.58,0.76])
# test2 = np.asarray([10.58,10.76])

# test1.reshape(1, -1)
# test2.reshape(1, -1)

# test_on_training = svm_clf.predict(X_matrix)
# print 'The prediction accuracy is:', sum(abs(Y - test_on_training))/len(Y)

# predict_results = (svm_clf.predict(X_matrix_test))
# print predict_results , sum(predict_results)


X_matrix_scale = np.zeros((len(amount_request), (FICO_1K + loan_1K + 6)))   
# Encode FICO into feature matrix
X_matrix_scale[:,0:FICO_1K] = train_encoded_FICO
# Encode Loan purpose into feature matrix
X_matrix_scale[:,FICO_1K:(loan_1K + FICO_1K)] = train_encoded_loan

for ii in range(0, len(amount_request)):
	X_matrix_scale[ii, (loan_1K + FICO_1K)] = (amount_request[ii] - min(amount_request)) / range_amount_request

for ii in range(0, len(interest_rate)):
	X_matrix_scale[ii, (loan_1K + FICO_1K + 1)] = (interest_rate[ii] - min(interest_rate)) / range_interest_rate

for ii in range(0, len(loan_length)):
	X_matrix_scale[ii, (loan_1K + FICO_1K + 2)] = (loan_length[ii] - min(loan_length)) / range_loan_length

for ii in range(0, len(month_pay)):
	X_matrix_scale[ii, (loan_1K + FICO_1K + 3)] = (month_pay[ii] - min(month_pay)) / range_month_pay

for ii in range(0, len(total_fund)):
	X_matrix_scale[ii, (loan_1K + FICO_1K + 4)] = (total_fund[ii] - min(total_fund)) / range_total_fund

for ii in range(0, len(DIT_ratio)):
	X_matrix_scale[ii, (loan_1K + FICO_1K + 5)] = (DIT_ratio[ii] - min(DIT_ratio)) / range_DIT_ratio					
#X_matrix_scale[:,(loan_1K + FICO_1K + 1)] = 
# X_matrix_scale[:,(loan_1K + FICO_1K + 2)] = train_encoded_loan
# X_matrix_scale[:,(loan_1K + FICO_1K + 3)] = train_encoded_loan
# X_matrix_scale[:,(loan_1K + FICO_1K + 4)] = train_encoded_loan
# X_matrix_scale[:,(loan_1K + FICO_1K + 5)] = train_encoded_loan

# svm_clf = svm.SVC(kernel='rbf', C = 1.0)
# # if you want a linear kernel clf = svm.SVC(kernel='linear', C = ??)
# # C = <number> is the relaxation penalty that you must choose (do not run the code with ??)
# svm_clf.fit(X_matrix_scale,decision)

# # test1 = np.asarray([0.58,0.76])
# # test2 = np.asarray([10.58,10.76])

# # test1.reshape(1, -1)
# # test2.reshape(1, -1)

# test_on_training = svm_clf.predict(X_matrix_scale)
# print 'The prediction accuracy is (scaled float):', sum(abs(Y - test_on_training))/len(Y)


X_matrix_nonscale = np.zeros((len(amount_request), (FICO_1K + loan_1K + 6)))   
# Encode FICO into feature matrix
X_matrix_nonscale[:,0:FICO_1K] = train_encoded_FICO
# Encode Loan purpose into feature matrix
X_matrix_nonscale[:,FICO_1K:(loan_1K + FICO_1K)] = train_encoded_loan

for ii in range(0, len(amount_request)):
	X_matrix_nonscale[ii, (loan_1K + FICO_1K)] = amount_request[ii]

for ii in range(0, len(interest_rate)):
	X_matrix_nonscale[ii, (loan_1K + FICO_1K + 1)] = interest_rate[ii] 

for ii in range(0, len(loan_length)):
	X_matrix_nonscale[ii, (loan_1K + FICO_1K + 2)] = loan_length[ii]

for ii in range(0, len(month_pay)):
	X_matrix_nonscale[ii, (loan_1K + FICO_1K + 3)] = month_pay[ii] 

for ii in range(0, len(total_fund)):
	X_matrix_nonscale[ii, (loan_1K + FICO_1K + 4)] = total_fund[ii] 

for ii in range(0, len(DIT_ratio)):
	X_matrix_nonscale[ii, (loan_1K + FICO_1K + 5)] = DIT_ratio[ii] 		
#X_matrix_scale[:,(loan_1K + FICO_1K + 1)] = 
# X_matrix_scale[:,(loan_1K + FICO_1K + 2)] = train_encoded_loan
# X_matrix_scale[:,(loan_1K + FICO_1K + 3)] = train_encoded_loan
# X_matrix_scale[:,(loan_1K + FICO_1K + 4)] = train_encoded_loan
# X_matrix_scale[:,(loan_1K + FICO_1K + 5)] = train_encoded_loan

# svm_clf = svm.SVC(kernel='rbf', C = 1.0)
# # if you want a linear kernel clf = svm.SVC(kernel='linear', C = ??)
# # C = <number> is the relaxation penalty that you must choose (do not run the code with ??)
# svm_clf.fit(X_matrix_nonscale,decision)

# # test1 = np.asarray([0.58,0.76])
# # test2 = np.asarray([10.58,10.76])

# # test1.reshape(1, -1)
# # test2.reshape(1, -1)

# test_on_training = svm_clf.predict(X_matrix_nonscale)
# print 'The prediction accuracy is (non scaled float):', sum(abs(Y - test_on_training))/len(Y)


# # Set up the initial W for update
# X_product = np.dot(X_matrix.transpose(), X_matrix)
# print X_product.shape
# X_product_inv = linalg.pinv(X_product) 
# W = np.dot(X_product_inv, X_matrix.transpose())
# W = np.dot(W, decision)
# print W.shape

# y = np.dot(W, X_matrix.transpose())
# print y

# y_decision = []
# sum_wrong = 0
# for ii in range(0, len(y)):
# 	if y[ii] > 0.5:
# 		sum_wrong = sum_wrong + abs(1 - decision[ii])
# 	else:
# 		sum_wrong = sum_wrong + abs(0 - decision[ii])

# print sum_wrong/len(decision)		
		

# def W_update(W_old, decision):
# 	Rnn = np.zeros(len(decision), len(decision))


def k_fold_train(fold_num, fold_idx, input_data, decision):
	fold_size = math.floor(len(decision)/fold_num)
	#exc_cnt = 0
	test_feature = []
	test_decision = []
	for ii in range(0, int(fold_size)):
		test_feature.append(input_data[(int(fold_size)*fold_idx)])
		test_decision.append(decision[(int(fold_size)*fold_idx)])	
		decision = np.delete(decision, (int(fold_size)*fold_idx), axis = 0)
		input_data = np.delete(input_data, (int(fold_size)*fold_idx), axis = 0)
		#exc_cnt += 1
	#print exc_cnt	
	# return_train = np.delete(input, input[fold_size*fold_idx:fold_size*(fold_idx + 1)], axis = 0)
	# return_train_decision = np.delete(decision, decision[1], axis = 0)
	return [input_data, decision, test_feature, test_decision]


# [return_train, return_train_decision, test_feature, test_decision] = k_fold_train(5, 1, X_matrix_scale, decision)
# print return_train.shape
# print return_train_decision.shape

# svm_clf.fit(return_train,return_train_decision)
# k_fold_test = svm_clf.predict(test_feature)
# print 'The prediction accuracy is (non scaled float), k-fold:', sum(abs(test_decision - k_fold_test))/len(test_decision)

# [return_train, return_train_decision, test_feature, test_decision] = k_fold_train(5, 2, X_matrix_scale, decision)
# print sum(test_decision)

fold_num = 10

## Feature Matrices obtained from previous parts:
# X_matrix: All 1-of-K binary coding matrix
# X_matrix_scale: Scaled floating features 
# X_matrix_nonscale: Non-scaled floating features 

## Classifiers:
# NBC_clf: Naive Bayes Classifier for Binary

for ii in range(0,fold_num):
	[return_train, return_train_decision, test_feature, test_decision] = k_fold_train(fold_num, ii, X_matrix, decision)
	# print return_train.shape
	# print return_train_decision.shape
	# print np.asarray(test_feature).shape
	# print np.asarray(test_decision).shape
	NBC_clf.fit(return_train,return_train_decision)
	k_fold_test = NBC_clf.predict(np.asarray(test_feature))
	#print 'The prediction accuracy is (non scaled float), k-fold:', sum(abs(test_decision - k_fold_test))/len(test_decision)
	tmp_TP = 0
	tmp_TN = 0
	tmp_FP = 0
	tmp_FN = 0
	for idx in range(0, len(k_fold_test)):
		if (k_fold_test[idx] == 1.0) and (decision[idx] == 1.0):
			tmp_TP += 1
		elif (k_fold_test[idx] == 1.0) and (decision[idx] == 0.0):
			tmp_FP += 1
		elif (k_fold_test[idx] == 0.0) and (decision[idx] == 0.0):
			tmp_TN += 1
		else:
			tmp_FN += 1
	# print tmp_TP, tmp_FP, tmp_TN, tmp_FN		
	tmp_precision = float(tmp_TP)/float(tmp_TP + tmp_FP)
	tmp_recall = float(tmp_TP)/float(tmp_TP + tmp_FN)
	tmp_F1 = float(2*tmp_recall*tmp_precision)/float(tmp_recall+tmp_precision)
	print 'F1 score for ', fold_num, 'folds, run ',ii, tmp_F1		


# changes = []
# start = 3482
# for ii in range(0, len(predict_results) + 1):
# 	if ii == 0:
# 		changes.append(['Record Number', 'Status'])
# 	else:
# 		changes.append([str(start+ii-1), str(int(predict_results[ii-1]))])


# with open('test_submit.csv', 'ab') as f:                                    
#     writer = csv.writer(f)                                                       
#     writer.writerows(changes)