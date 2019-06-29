import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
'''  This example is a very simple image classifier, much cruder than the MHIST example as it uses a representation in
 four pixels to make it as simple as possible as a learning exercise.

Consider these as four characters possible in a 2x2 matrix of pixels
0 = Blank 1 = Active

00 10 01 11
00 01 10 11

let us use numpy reshape to reshape to one dimension for input like the MNIST examples.

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
#quit()


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

input_layer = 4
output_layer = 4
hidden_layer = 2
# initialize weights randomly with mean 0
# syn0 weights for input layer with input_layers to hidden_layers dimension
syn0 = 2*np.random.random((input_layer,hidden_layer)) - 1

# One hidden_layer bias
b0 = 2.0*np.random.random((1,hidden_layer)) - 1

# Syn1 is output layer with hidden_layers and output_layers dimension
syn1 = 2.0*np.random.random((hidden_layer,output_layer)) - 1
# One output_layer bias
b1 = 2.0*np.random.random((1,output_layer)) - 1



for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0) + b0)
    l2 = nonlin(np.dot(l1,syn1) + b1) 

    # how much did we miss?
    l2_error = y - l2
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l2_delta = l2_error * nonlin(l2,True)


    l1_error = np.dot(l2_delta,syn1.T)

    l1_delta = l1_error * nonlin(l1,True)
  # . T
    # update weights and biases
    syn1 += np.dot(l1.T,l2_delta)
    syn0 += np.dot(l0.T,l1_delta)
    b1 += np.sum(l2_delta,axis=0,keepdims=True)	
    b0 += np.sum(l1_delta,axis=0,keepdims=True) 

print("Output After Training:")
print(l2)
lt = (l2 > 0.5) * l2
print("Remove low values to make the output cleaner.")
print(lt)



