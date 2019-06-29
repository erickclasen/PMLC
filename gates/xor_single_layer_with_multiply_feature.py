import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
# For the logic function  we will fill the first row with 1 as a bias. Without it and just trying to input 2 columns.
# It will just go to 0.5 when the input is 0 0.
print("Simple Neural Network: XOR function. Linear Version.")
# any ML program that can compute XOR is capable in theory of doing anything.

X = np.array([  [1,0,0,0],
                [1,0,0,1],
                [1,0,1,0],
                [1,1,1,1] ])
    
# output dataset            
y = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((4,1)) - 1

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

print("Original input features")
print(l0)
print("Targets y")
print(y)
print("Weights")
print(syn0)


print("Output After Training:")
print(l1)


# Manual multiply to understand how the vector mul works.

print("Do an example explicitly with regular Algebra to show how it works")
print(syn0[0],syn0[1],syn0[2],syn0[3])
test = syn0[0] * 1 + syn0[1] * 1 + syn0[2] * 1 + syn0[3] * 1
#test = nonlin(test)
# Sigmoid non-linearity
test = 1/(1+np.exp(-test))

print(test)
