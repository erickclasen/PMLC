''' Simple Almost Machine Learning Linear Solver. '''

# Output, input and initial m
y = 5
x = 2
m = 0

# Hyper parameters
# Learning rate and number of loops.
l = 0.25
n = 20

print("i,m,y1")

for i in range(0,n):
	# Predict
	y1 = m*x
	print(i,m,y1)

	# Update weight, by one step of the learning rate.
	m = m + l

       # If the error is less than zero, going the wrong direction, jump back 2 learning rate units.
	if y - y1 < 0:
		m = m - 2*l           		
