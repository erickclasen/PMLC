# Super Basic Machine Learning using linear regression.
# Linear Regression using Gradient Descent

# Resources
# https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
# https://towardsdatascience.com/linear-regression-using-gradient-descent-in-10-lines-of-code-642f995339c0


''' Update weights add a bias, DO NOT activate as this is linear function fit'''
def update_weights(xa,xb, y, weighta,weightb, bias, learning_rate):
    weight_deriva = 0
    weight_derivb = 0
    bias_deriv = 0
    N = len(y)

    for i in range(N):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriva += -1*xa[i] * (y[i] - (weighta*xa[i] + weightb*xb[i] + bias))
        weight_derivb += -1*xb[i] * (y[i] - (weighta*xa[i] + weightb*xb[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -1*(y[i] - (weighta*xa[i] + weightb*xb[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weighta -= (weight_deriva / N) * learning_rate
    weightb -= (weight_derivb / N) * learning_rate

    bias -= (bias_deriv / N) * learning_rate

    return weighta, weightb, bias

''' Cost is for our observation of how well it is closing in on a target fit. Uses mean squared error.
    We could if we wanted to break out of the loop when the cost gets below an acceptable threshold. '''
def cost_function(xa,xb, y, weighta, weightb, bias):
    N = len(xa)
    total_error = 0.0
    for i in range(N):
        total_error += (y[i] - (weighta*xa[i] + weightb*xb[i] + bias))**2
    return total_error / N

''' Show a simple chart of the x and y from the lists and the predicted y and the plain error y- predicted y. '''
def results(weighta,weightb,bias,xa,xb,y):

	# Print out a simple heading.
        print("\n\nxa[i],xb[i],y[i],y_pred,error")

	for i in range(len(y)):
		y_pred = weighta * xa[i] + weightb * xb[i] + bias
                error = y[i]-y_pred
		print(xa[i],xb[i],y[i],y_pred,error)

''' Training routine. Just runs through a set of interations of epochs length and updates weight and bias.
    For educational purposes prints out the results in every loop. In practice this would be overkill. '''
def linear_regression_train(xa,xb, y, weighta=0, weightb=0, bias=0, epochs=1000, learning_rate=0.0001,print_status=True):
        # Print out a simple heading for the data.
	if print_status:
        	print("iter,epochs,cost,weighta,weightb,bias")

	# Run through the update weights and cost function code until we are done!

	for iter in range(epochs):

        	weighta,weightb,bias = update_weights(xa, xb, y,weighta, weightb, bias, learning_rate)
        	cost = cost_function(xa,xb, y, weighta, weightb, bias)
        	# Show what is happening
        	if print_status:
			print(iter,epochs,cost,weighta,weightb,bias)

	return weighta,weightb,bias,cost

def optimize_learning_rate(xa,xb,y):
	best_cost = 9999999999
	best_learning_rate = 0

	# Hyperparams, don't need a lot of epochs to make this fit and rate can be a bit higher than 0.001 which it started out with.
	# Train the model

	for learning_rate_train in range(1,10000,1):
        	weighta,weightb,bias,cost = linear_regression_train(xa, xb, y, epochs=10, learning_rate=float(learning_rate_train)/10000,print_status=False)
        	# Remember the best cost.
        	if cost < best_cost:
                	best_cost = cost
                	best_learning_rate = float(learning_rate_train)/10000

        print(best_cost,best_learning_rate) 
	quit()


''' Main Code for Super Simple Machine Learning '''
# Can't get any more linear than this, both x and y are the same thing.
xa = [1,2,3,4,5,6,7,8,9,10]
xb = [1,2,3,4,5,6,7,8,9,10]

y = [1,2,3,4,5,6,7,8,9,10]
#y = [2,4,6,8,10,12,14,16,18,20]

#optimize_learning_rate(xa,xb,y)

#Train the model
weighta,weightb,bias,cost = linear_regression_train(xa, xb, y, epochs=10, learning_rate=0.0072)

# Show a chart of the results.
results(weighta, weightb, bias, xa, xb, y)

