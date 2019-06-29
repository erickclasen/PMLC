import numpy as np

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

''' This is just here to see how it would work, an experiment. '''
def de_normalize(targets,train_targets):
    '''
    features     -   (10, 3)
    features.T   -   (3, 10)

    We transpose the input matrix, swapping
    cols and rows to make vector math easier

    To de-normalize we have to first normalize off the
    targets that we trained with and then use the fmean and frange
    to de-normalize to result targets to scale the predictor results. 	

    '''

    for train_targets in train_targets.T:
        fmean = np.mean(train_targets)
        frange = np.amax(train_targets) - np.amin(train_targets)

        #Vector Multiplication First. Opposite order of normalization.
        targets *= frange

        #Vector Addition Second. Opposite order of normalization.
        targets += fmean


    return targets 


def predict(features, weights):
  '''
  features - (10, 3)
  weights - (3, 1)
  predictions - (10,1)
  '''
  predictions = np.dot(features, weights)
  return predictions


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

# Weights
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

# Features as lists
x1 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
x2 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
x3 = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
#bias = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#bias = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

# Convert features to array and transpose
features = np.array([
#		bias,
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

# Put a bias of one into the first row of features. This will work with the biasw (weight) for a bias.
bias = np.ones(shape=(len(features),1))
features = np.append(bias, features, axis=1)
print("Normalized Features wih the bias column added.")
print(features)

# y is the target, must be the same length and shape as all the x's, so transpose it.
y = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
targets = np.array([
		y
	])


# Traspose the targets.
targets = targets.T
print("orig targets transposed")
print(targets)

# Set a learning rate
lr = 0.01
epochs = 2000

print("Start...")
# Epochs = loops. Run the update weights function over and over as long as needed for convergence.
# Occasionally, once per 1000 loops display the loss to get an idea how it is progressing.
for i in range(epochs):
	#Run the weights update once to backpropigate.
	update_weights_vectorized(features, targets, weights, lr)
	if i%1000 ==0:
		print("Iteration:",i)	
		print("Updated Weights")
		print(weights)
		print("cost:",cost_function(features, targets, weights))
		print("")

print("...Done!")
# Run a prediction to show the results. The targets will still be in a normalized state.
predictions = predict(features, weights)


print("predictions",predictions)
print("error: targets - predictions",targets - predictions)



