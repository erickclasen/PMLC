# Super Basic Machine Learning using linear regression.
# Linear Regression using Gradient Descent

# Resources
# https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
# https://towardsdatascience.com/linear-regression-using-gradient-descent-in-10-lines-of-code-642f995339c0


''' Update weights add a bias, DO NOT activate as this is linear function fit'''
def update_weights(x, y, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    N = len(y)

    for i in range(N):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2*x[i] * (y[i] - (weight*x[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2*(y[i] - (weight*x[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (weight_deriv / N) * learning_rate
    bias -= (bias_deriv / N) * learning_rate

    return weight, bias

''' Cost is for our observation of how well it is closing in on a target fit. Uses mean squared error.
    We could if we wanted to break out of the loop when the cost gets below an acceptable threshold. '''
def cost_function(x, y, weight, bias):
    N = len(x)
    total_error = 0.0
    for i in range(N):
        total_error += (y[i] - (weight*x[i] + bias))**2
    return total_error / N

''' Show a simple chart of the x and y from the lists and the predicted y and the plain error y- predicted y. '''
def results(weight,bias,x,y):

	# Print out a simple heading.
        print("\n\nx[i],y[i],y_pred,error")

	for i in range(len(y)):
		y_pred = weight * x[i] + bias
                error = y[i]-y_pred
		print(x[i],y[i],y_pred,error)

''' Training routine. Just runs through a set of interations of epochs length and updates weight and bias.
    For educational purposes prints out the results in every loop. In practice this would be overkill. '''
def linear_regression_train(x, y, weight=0, bias=0, epochs=1000, learning_rate=0.0001,print_status=True):
        # Print out a simple heading for the data.
        if print_status:
        	print("iter,epochs,cost,weight,bias")

	# Run through the update weights and cost function code until we are done!

	for iter in range(epochs):

        	weight,bias = update_weights(x, y, weight, bias, learning_rate)
        	cost = cost_function(x, y, weight, bias)
        	# Show what is happening
		if print_status:
        		print(iter,epochs,cost,weight,bias)

	return weight,bias,cost

''' Main Code for Super Simple Machine Learning '''
# Can't get any more linear than this, both x and y are the same thing.
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]

best_cost = 9999999999
best_learning_rate = 0

# Hyperparams, don't need a lot of epochs to make this fit and rate can be a bit higher than 0.001 which it started out with.
# Train the model
for learning_rate_train in range(1,10000,1):
	weight,bias,cost = linear_regression_train(x, y, epochs=10, learning_rate=float(learning_rate_train)/10000,print_status=False)
        # Remember the best cost.
	if cost < best_cost:
		best_cost = cost
		best_learning_rate = float(learning_rate_train)/10000

weight,bias,cost = linear_regression_train(x, y, epochs=10, learning_rate=best_learning_rate)

# Show a chart of the results.
results(weight,bias,x,y)
print("Best Cost and Optimized Learning Rate",best_cost,best_learning_rate)

