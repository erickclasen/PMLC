'''
http://neuralnetworksanddeeplearning.com/chap1.html
There is a way of determining the bitwise representation of a digit by adding an extra layer to the three-layer network above. The extra layer converts the output from the previous layer into a binary representation, as illustrated in the figure below. Find a set of weights and biases for the new output layer. Assume that the first 3 layers of neurons are such that the correct output in the third layer (i.e., the old output layer) has activation at least 0.99, and incorrect outputs have activation less than 0.01. 

Crossenthropy code from https://towardsdatascience.com/neural-net-from-scratch-using-numpy-71a31f6e3675
mse from https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy
'''

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def unit_step(v):
        """ Heavyside Step function. v must be a scalar """
        if v >= 0.5:
                return 1
        else:
                return 0

'''  This example is a very simple image classifier, much cruder than the MHIST example as it uses a representation in
 four pixels to make it as simple as possible as a learning exercise. 

Consider these as four characters possible in a 2x2 matrix of pixels
0 = Blank 1 = Active

00 10 01 11
00 01 10 11

let us use numpy reshape to reshape to one dimension for input like the MNIST examples do.

1. It is possible to classify these in this simple example with a 1 layer forward NN.

'''
a = np.array([  [0,0],
                [0,0] ])

b = np.array([  [1,0],
                [0,1] ])

c = np.array([  [0,1],
                [1,0] ])

d = np.array([  [1,1],
                [1,1] ])

stacked_array = np.vstack(((((a,b,c,d)))))
print("Stacked Array")
print(stacked_array)
print("Shape to 4,4")
X = stacked_array.reshape(4,4)
print(X)


# output dataset            
#y = np.array([[0,1,1,0]]).T
# This is the classified output, as in which 'bin' does the digit fall.
y = np.array([   [1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]  ])
    

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# Define the layer width sizes here.
input_layer = 4
output_layer = 4

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((input_layer,output_layer)) - 1

# One output is one bias
b0 = 2.0*np.random.random((1,output_layer)) - 1

for iter in range(10000):

    # forward propagation
    l0 = X
    #l1 = nonlin(np.dot(l0,syn0))
    l1 = nonlin(np.dot(l0,syn0) + b0)

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1

    # Use the derivative of the sigmoid at the l1 values to get the slope at that position of the curve.
    # This will allow the calculation of the partial derivative, a small nudge, delta of the l1 value as l1_delta
    # which is the error multiplied by the derivative. Then we nudge all the weights and biases a bit based on this
    # small delta. This is basically backpropigation, using the error and derivatives of the activation function to
    # nudge the weights slightly in the direction of lowering the error.
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    b0 += np.sum(l1_delta,axis=0,keepdims=True) 

    #Show progress of the decresing loss, do this with a few types of loss calculations.
    if iter % 1000 == 0:
        sum_of_squares = sum(sum(l1_error*l1_error))
        # Compute the cross-entropy cost
        m = 4
        logprobs = np.multiply(np.log(l1), y) + np.multiply((1 - y), np.log(1 - l1))
        cost = - np.sum(logprobs) / m
        
        # Mean squared error
        mse = (np.square(l1 - y)).mean(axis=None)
        print(iter,cost,sum_of_squares,mse)

        

print("Output After Training:")
print(l1)
print("Clip all the low values to make it easier to read...")
lt = (l1 > 0.5) * l1
print(lt)

print("Weights and Biases--------------------------------------")
print("weights")
print(syn0)
print('biases')
print(b0)


print("Try out some junk values that the classifier has not seen before.")

a = np.array([  [1,0],
                [0,0] ])

b = np.array([  [1,1],
                [0,1] ])

c = np.array([  [1,1],
                [1,0] ])

d = np.array([  [1,1],
                [1,0] ])

stacked_array = np.vstack(((((a,b,c,d)))))
#print("Stacked Array")
#print(stacked_array)
#print("Shape to 4,4")
X = stacked_array.reshape(4,4)

#X = np.array([ [1,0,0,0] ])
print(X)

# forward propagation
l0 = X
l1 = nonlin(np.dot(l0,syn0) + b0)
print(l1)
print("Clip all the low values to make it easier to read...")
lt = (l1 > 0.5) * l1
print(lt)

