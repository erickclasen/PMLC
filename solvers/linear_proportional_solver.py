''' Linear Proportional Solver, has all the basic elements of machine learning. '''
# Output, input and initial m
y = 5
x = 2
m = 0.5

# Hyper parameters
# Learning rate and number of loops
l = 0.25
n = 10

print("i,m,y1")

for i in range(0,n):
	# Predict
	y1 = m*x
	print(i,m,y1)

	# Update weight
	m = m + l*(y-y1)
