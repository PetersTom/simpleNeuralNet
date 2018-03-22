import numpy as np
from sklearn import preprocessing
import pandas as pd
import sys


# The sigmoid function
def sigmoid(x, deriv=False):
    if (deriv):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


# x = np.array([[0, 0, 1],
#              [0, 1, 1],
#              [1, 0, 1],
#              [1, 1, 1]])
#
# y = np.array([[0, 1, 1, 0]]).T

# Import data and scale it
#
# scale values from to values between -1 and 1
min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
df = pd.read_csv("data.csv", header=0)
# clean up data
df.columns = ["grade1", "grade2", "label"]
# remove the trailing ;
x = df["label"].map(lambda x: float(x.rstrip(';')))

# formats the input data into two arrays, one of independent variables
# and one of the dependant variable
x = df[["grade1", "grade2"]]
x = np.array(x)
x = min_max_scaler.fit_transform(x)
y = df["label"].map(lambda x: float(x.rstrip(';')))
y = np.array([y]).T

np.random.seed(1)

syn0 = 2*np.random.random((2, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

previous_error = sys.maxsize
# actual training
for i in range(sys.maxsize):
    l1 = sigmoid(x.dot(syn0))
    l2 = sigmoid(l1.dot(syn1))

    l2_error = y - l2

    current_error = np.mean(np.abs(l2_error))
    if np.abs(previous_error - current_error) < 10**-5:
        print("converged on iteration " + str(i))
        print("previous error: " + str(previous_error))
        print("current error: " + str(current_error))
        break
    previous_error = current_error
    if (i % 100000) == 0:
        print("Error:" + str(current_error))

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_error.dot(syn1.T)

    l1_delta = l1_error * sigmoid(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += x.T.dot(l1_delta)
