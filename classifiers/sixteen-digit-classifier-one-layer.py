'''
http://neuralnetworksanddeeplearning.com/chap1.html
There is a way of determining the bitwise representation of a digit by adding an extra layer to the three-layer network above. The extra layer converts the output from the previous layer into a binary representation, as illustrated in the figure below. Find a set of weights and biases for the new output layer. Assume that the first 3 layers of neurons are such that the correct output in the third layer (i.e., the old output layer) has activation at least 0.99, and incorrect outputs have activation less than 0.01. 

Crossenthropy code from https://towardsdatascience.com/neural-net-from-scratch-using-numpy-71a31f6e3675
mse from https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy

https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
'''

import numpy as np
import json

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

there are 12 more possible...

10 01 00 10
00 00 01 00

11 00 10 01
00 11 10 01

01 10 11 01
11 11 10 11

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

stacked_array1 = np.vstack(((((a,b,c,d)))))


'''
10 01 00 10
00 00 01 00

'''
e = np.array([  [1,0],
                [0,0] ])

f = np.array([  [0,1],
                [0,0] ])

g = np.array([  [0,0],
                [0,1] ])

h = np.array([  [1,0],
                [0,0] ])

stacked_array2 = np.vstack(((((e,f,g,h)))))

'''
11 00 10 01
00 11 10 01
'''
i = np.array([  [1,1],
                [0,0] ])

j = np.array([  [0,0],
                [1,1] ])

k = np.array([  [1,0],
                [1,0] ])

l = np.array([  [0,1],
                [0,1] ])

stacked_array3 = np.vstack(((((i,j,k,l)))))

'''
01 10 11 01
11 11 10 11
'''
m = np.array([  [0,1],
                [1,1] ])

n = np.array([  [1,0],
                [1,1] ])

o = np.array([  [1,1],
                [1,0] ])

p = np.array([  [0,1],
                [1,1] ])

stacked_array4 = np.vstack(((((m,n,o,p)))))


stacked_array = np.vstack(((((stacked_array1,stacked_array2,stacked_array3,stacked_array4)))))
print("Stacked Array")
print(stacked_array)
print("Shape to 16,4")
X = stacked_array.reshape(16,4)
print(X)



X = np.array([   [0,0,0,0],
                 [0,0,0,1],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,1,0,0],
                 [0,1,0,1],
                 [0,1,1,0],
                 [0,1,1,1],
                 [1,0,0,0],
                 [1,0,0,1],
                 [1,0,1,0],
                 [1,0,1,1],
                 [1,1,0,0],
                 [1,1,0,1],
                 [1,1,1,0],
                 [1,1,1,1]  ])


# output dataset            
#y = np.array([[0,1,1,0]]).T
# This is the classified output, as in which 'bin' does the digit fall.
y = np.array([   [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]  ])
print()
print(y)    

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# Define the layer width sizes here.
input_layer = 4
output_layer = 16

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


filename = 'weights.json'
syn0_list = syn0.tolist()
b0_list = b0.tolist()
with open(filename, 'w') as f_obj:
                    json.dump(syn0_list,f_obj)

filename = "biases.json"
with open(filename, 'w') as f_obj:
                    json.dump(b0_list,f_obj)


filename = 'weights.json'

# If the file exists read it in, if not just create it on the first pass through.
try:
        with open(filename) as f_obj:
                 w_list = json.load(f_obj)

except IOError:
        print("Missing File")
        quit()

filename = "biases.json"

# If the file exists read it in, if not just create it on the first pass through.
try:
        with open(filename) as f_obj:
                 b_list = json.load(f_obj)

except IOError:
        print("Missing File")
        quit()


weights = np.array(w_list)
biases = np.array(b_list)

print(weights,biases)

#quit()
print("Try out a value using the trained weights and biases.")

X = np.array([  [0,0,0,1] ] )

# forward propagation
l0 = X
l1 = nonlin(np.dot(l0,weights) + biases)
print(l1)
print("Clip all the low values to make it easier to read...")
lt = (l1 > 0.5) * l1
print(lt)

