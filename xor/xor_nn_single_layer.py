import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

print("XOR: With a hidden layer and the non-linear activation function on the output layer.")
print("Fails!")
'''
2 inputs 1 output

l0 is the input layer values, aka X
l1 is the hidden layer values

syn0 synapse 0 is the weight matrix for the output layer

b0 is the bias for the output layer.

X is the input matrix of features.
Y is the target to be learned.

'''



    
# input dataset
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1] ])
    
# output dataset            
y = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# How wide are the layers, how many neurons per layer?
input_layer = 2
output_layer = 1

# initialize weights randomly with mean 0
# syn0 weights for input layer with input_layers to output_layers dimension
syn0 = 2*np.random.random((input_layer,output_layer)) - 1

# One output_layer bias
b0 = 2.0*np.random.random((1,output_layer)) - 1


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

