import csv
import pandas as pd
import numpy as np 
#from scikits.learn import svm
import networkx as nx

with open('Bank_Data_Train.csv','rb') as training:
	training_reader = csv.reader(training)
	#for train_row in training_reader:
		#print train_row


with open('Bank_Data_Test.csv','rb') as testing:
	testing_reader = csv.reader(testing)
	#for test_row in testing_reader:
		#print test_row		

def data_processing(filename):
    df = pd.read_csv(filename)
   # remember: always use an intercept for SVM and Logistic regression
    df['intercept'] = 1.0
    return (df)

train = data_processing('Bank_Data_Train.csv')
test = data_processing('Bank_Data_Test.csv')

encoded = pd.get_dummies(pd.concat([train['FICO Range'],test['FICO Range']], axis=0),\
prefix='FICO Range', dummy_na=True)
train_rows = train.shape[0]
train_encoded = encoded.iloc[:train_rows, :]
test_encoded = encoded.iloc[train_rows:, :]

print(train_encoded.shape)
print(test_encoded.shape)

print(type(train_encoded))

cnt = 0
for row in train_encoded:
	print(row)
	cnt += 1
print(cnt)

k_vec =  train_encoded['FICO Range_660-664']
for element in k_vec:
	print(element)
