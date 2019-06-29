''' https://iamtrask.github.io//2015/07/12/basic-python-network/ '''
import numpy as np
import math
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def error(l1_error):
    cum_sum = 0

    for n in range (0,len(l1_error)):
        cum_sum += l1_error[n] * l1_error[n]

    return math.sqrt(cum_sum)

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

print("Synapses")
print(syn0)

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

    if iter % 1000 == 0 or iter == 1:
        print(iter)
        print(l1)
        print("error:",error(l1_error))

print( "Output After Training:")
print(l1)
print("error:",error(l1_error))
print("Synapses")
print(syn0)

print("Validate on Unknown")
# forward propagation
X = np.array([  [1,0,0],
                [1,1,0],
                [1,0,1],
                [1,1,1] ])
print(X)
l0 = X
l1 = nonlin(np.dot(l0,syn0))
print(l1)


