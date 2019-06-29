import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

print("XOR: With a one layer and no activation function.")
print("The trick is to use feature engineering and multiply both the X inputs together and use that as a third input.")
'''
2 inputs 1 output
1 added input feature

l0 is the input layer values, aka X
l1 is the hidden layer values

syn0 synapse 0 is the weight matrix for the output layer

b0 is the bias for the output layer.

X is the input matrix of features. Column 1 will be the new feature and is initialized to all zeros.
Y is the target to be learned.

'''



    
# input dataset, the leftmost column is initialized to zero as this will contain the new feature of X[1]*X[2].
X = np.array([  [0,0,0],
                [0,0,1],
                [0,1,0],
                [0,1,1] ])
    
# output dataset            
y = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# How wide are the layers, how many neurons per layer?
# This input layer now has three positions, the original two inputs for X and the feature engineering of 
# both original X's.
input_layer = 3
output_layer = 1

# initialize weights randomly with mean 0
# syn0 weights for input layer with input_layers to output_layers dimension
syn0 = 2*np.random.random((input_layer,output_layer)) - 1

# One output_layer bias
b0 = 2.0*np.random.random((1,output_layer)) - 1


# First, before running the loop do some feature engineering and multiply the 2 X inputs
# together and store in the first column.
for n in range(0,4):
    X[n][0] = X[n][1] * X[n][2]

# Showing this explicitly with numbered positions, to show what the results are.
print(X[0][0],X[0][1],X[0][2])
print(X[1][0],X[1][1],X[1][2])
print(X[2][0],X[2][1],X[2][2])
print(X[3][0],X[3][1],X[3][2])


for iter in range(10000):

    # forward propagation
    l0 = X

    l1 = nonlin(np.dot(l0,syn0) + b0)

    # how much did we miss?
    l1_error = y - l1
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l2
    l1_delta = l1_error * nonlin(l1,True)

    # update weights and biases
    syn0 += np.dot(l0.T,l1_delta)
    b0 += np.sum(l1_delta,axis=0,keepdims=True) 

print("Output After Training:")
print(l1)

