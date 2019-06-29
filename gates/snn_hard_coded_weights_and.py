import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
# For the logic function  we will fill the first row with 1 as a bias. Without it and just trying to input 2 columns the results are poor.

print("Simple Neural Network: AND function. ")

X = np.array([  [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1] ])
    
# Use the prelearned syn0, with 1 bias on the top and two weights, which are the same.
syn0 = np.array([  [-12.11456643],
 		   [  8.01873355],
 		   [  8.01873355] ])


# Our input to the neural net for the test.
X = np.array([  [1,0,0] ] )

print("The input, a bias of 1 and two zeroes",X)
l0 = X
l1 = nonlin(np.dot(l0,syn0))

# A test output

print("The output should be a low value for false.",l1)
