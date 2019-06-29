import numpy as np

def predict(features, weights):
  '''
  features - (10, 3)
  weights - (3, 1)
  predictions - (10,1)
  '''
  predictions = np.dot(features, weights)
  return predictions

def cost_function(features, targets, weights):
    '''
    features:(10,3)
    targets: (10,1)
    weights:(3,1)
    returns average squared error among predictions
    '''
    N = len(targets)

    predictions = predict(features, weights)

    # Matrix math lets use do this without looping
    sq_error = (predictions - targets)**2

    # Return average squared error among predictions
    return 1.0/(2*N) * sq_error.sum()


# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

X = np.array([  [0,0,0],
                [0,0,1],
                [0,1,0],
                [0,1,1],
                [1,0,0],
                [1,0,1],
                [1,1,0],                
                [1,1,1]  ])

# One hot vectors
X = np.array([  [0,0,1],
                [0,1,0],
                [1,0,0] ])


    
# output dataset            
y = np.array([[0,0,1,1]]).T
y = np.array([[0,0,1,1,1,1,1,1]]).T
y = np.array([[0,1,2,3,4,5,6,7]]).T
y = np.array([[0,1,0]]).T

print(X)
print("")
print(y)
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

print("weights")
print(syn0)
print("")
for iter in xrange(100000):

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

print "Output After Training:"
print l1
print("")
print("Predictions")
print(nonlin(np.dot(l0,syn0)))
print("Error")
print(l1_error)

# Not right???
#print(cost_function(l0,y,syn0))

X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# Unseen data

X = np.array([  [0,0,0],
                [0,0,1],
                [0,1,0],
                [0,1,1],
                [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1]  ])

print("Test...")
print(X)
print("Pred...")
print(nonlin(np.dot(X,syn0)))

