'''
http://neuralnetworksanddeeplearning.com/chap1.html
There is a way of determining the bitwise representation of a digit by adding an extra layer to the three-layer network above. The extra layer converts the output from the previous layer into a binary representation, as illustrated in the figure below. Find a set of weights and biases for the new output layer. Assume that the first 3 layers of neurons are such that the correct output in the third layer (i.e., the old output layer) has activation at least 0.99, and incorrect outputs have activation less than 0.01. 
'''

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

    
# input dataset
X = np.array([  [0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1] ])
    
# output dataset            
y = np.array([   [0,0,0,0],
		 [0,0,0,1],
		 [0,0,1,0],
		 [0,0,1,1],
                 [0,1,0,0],
                 [0,1,0,1],
                 [0,1,1,0],
                 [0,1,1,1],
                 [1,0,0,0],
                 [1,0,0,1]  ])



# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

'''
  features - (10, 3)
  weights - (3, 1)
  predictions - (10,1)
'''

input_layer = 10
output_layer = 4

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((input_layer,output_layer)) - 1

# One output is one bias
b0 = 2.0*np.random.random((1,output_layer)) - 1

for iter in xrange(100000):

    # forward propagation
    l0 = X
    #l1 = nonlin(np.dot(l0,syn0))
    l1 = nonlin(np.dot(l0,syn0) + b0)

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    b0 += np.sum(l1_delta,axis=0,keepdims=True) 

    #Show progress
    if iter % 10000 == 0:
        # Print the iteration number and the cost function, sum of the squares of the error.
        print(iter,np.square(l1_error).sum())

	

print "Output After Training:"
print l1
print("Clip all the low values to make it easier to read...")
lt = (l1 > 0.5) * l1
print lt

# Beware of the pointer issue. For production code is l1 is to be used again make a copy first!
#s = copy.deepcopy(l1)

# Apply a step function, if > 0.5 snap to 1 if < 0.5 snap to 0.
l1[l1 >= 0.5] = 1

l1[l1 < 0.5] = 0
print("Step function applied")
print(l1)


print("Weights and Biases--------------------------------------")
print("weights")
print(syn0)
print('biases')
print(b0)

