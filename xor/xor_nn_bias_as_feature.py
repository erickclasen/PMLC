import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

print("XOR: Solved with a hidden layer and the non-linear activation function on both hidden and output layers.")
print("Trying the bias as an input feature, still no good.")

 
'''
2 inputs 1 output

l0 is the input layer values, aka X
l1 is the hidden layer values
l2 is the output layer values

syn0 synapse 0 is the weight matrix for the hidden layer
syn1 synapse 1 is the weight matrix for the output layer

b0 is the bias for the hidden layer.
b1 is the bias for the output layer.

X is the input matrix of features.
Y is the target to be learned.

'''



    
# input dataset
X = np.array([  [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# How wide are the layers, how many neurons per layer?
input_layer = 3
output_layer = 1
hidden_layer = 3

# initialize weights randomly with mean 0
# syn0 weights for input layer with input_layers to hidden_layers dimension
syn0 = 2*np.random.random((input_layer,hidden_layer)) - 1

# One hidden_layer bias
#b0 = 2.0*np.random.random((1,hidden_layer)) - 1
b0 = 0

# Syn1 is output layer with hidden_layers and output_layers dimension
syn1 = 2.0*np.random.random((hidden_layer,output_layer)) - 1
# One output_layer bias
#b1 = 2.0*np.random.random((1,output_layer)) - 1
b1 = 0

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0) + b0)
    l2 = nonlin(np.dot(l1,syn1) + b1) 

    # how much did we miss?
    l2_error = y - l2
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l2
    l2_delta = l2_error * nonlin(l2,True)

    # Ditto for the hidden layer, passing back the l2_delta.
    l1_error = np.dot(l2_delta,syn1.T)

    l1_delta = l1_error * nonlin(l1,True)

    # update weights and biases
    syn1 += np.dot(l1.T,l2_delta)
    syn0 += np.dot(l0.T,l1_delta)
    #b1 += np.sum(l2_delta,axis=0,keepdims=True) 
    #b0 += np.sum(l1_delta,axis=0,keepdims=True) 

print("Output After Training:")
print(l2)

