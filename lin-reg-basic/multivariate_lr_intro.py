''' This is just an intro and prints out the arrays as they are set upand then runs one update to the weights. '''
import numpy as np

''' When there are a lot of features and they have a wide range, it is best to normalize them by scling down by the
    max range and then subtracting the mean from them. Effectively rescaling them in a linear way and removing the
    mean centers the features around zero.'''
def normalize(features):
    '''
    features     -   (10, 3)
    features.T   -   (3, 10)

    We transpose the input matrix, swapping
    cols and rows to make vector math easier
    '''

    for feature in features.T:
        fmean = np.mean(feature)
        frange = np.amax(feature) - np.amin(feature)
        
        #Vector Subtraction
        feature -= fmean

        #Vector Division
        feature /= frange

    return features


''' Prediction is just a y= mx+b type of calculation. Except since we are dealing with vectors, we will use matrix math.
    The prediction (y) is the dot product of the weights and features.
    Also the bias is now working against the column of values of 1.0 in the first row of features, so it is not
    seperately listed. '''
def predict(features, weights):
  '''
  features - (10, 3)
  weights - (3, 1)
  predictions - (10,1)
  '''
  predictions = np.dot(features, weights)
  return predictions

''' Update the weights using a simple back-propigation. Getting the error from the predictions and then a 
    partial derivative then average by dividing by feature length and use this value to update weights. '''
def update_weights_vectorized(X, targets, weights, lr):
    '''
    gradient = X.T * (predictions - targets) / N
    X: (10, 3)
    Targets: (10, 1)
    Weights: (3, 1)
    '''
    N = len(X)

    #1 - Get Predictions
    predictions = predict(X, weights)

    #2 - Calculate error/loss
    error = targets - predictions

    #3 Transpose features from (10, 3) to (3, 10)
    # So we can multiply w the (10,1)  error matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(-X.T,  error)

    #4 Take the average error derivative for each feature
    gradient /= N

    #5 - Multiply the gradient by our learning rate
    gradient *= lr

    #6 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights

''' Cost function for our reference. Makes a prediction and calculates the MSE, Mean Squared Error. '''
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


''' Main Code '''

# Weights, init to zero
W1 = 0.0
W2 = 0.0
W3 = 0.0
biasw = 0.0

# Convert to an array (no transpose)
weights = np.array([
    [biasw],
    [W1],
    [W2],
    [W3]

])

print("weights")
print(weights)

# Features start out as lists
x1 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
x2 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
x3 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

# Convert features to array and transpose
features = np.array([
		x1,
		x2,
		x3
		
	])
features = features.T

print("features")
print(features)

# Normalize the features to bring them into a -1 to 1 range,centered around the mean
    #1 Subtract the mean of the column (mean normalization)
    #2 Divide by the range of the column (feature scaling)
features = normalize(features)
bias = np.ones(shape=(len(features),1))
features = np.append(bias, features, axis=1)
print("Normalized Features")
print(features)

# y is the target, must be the same length and shape as all the x's, so transpose it.
y = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
targets = np.array([
		y
	])
targets = targets.T
# Don't normalize the targets.
print("Targets")
print(targets)


# Set a learning rate
lr = 0.001
#Run the weights update once to show that they update successfully.
update_weights_vectorized(features, targets, weights, lr)
print("Updated Weights, starting to move off of zero")
print(weights)

# Cost after one run through.
print("cost:",cost_function(features, targets, weights))

# Prediction afte rone run through.
print("One shot prediction, starting to move off of zero")
print(predict(features, weights))

