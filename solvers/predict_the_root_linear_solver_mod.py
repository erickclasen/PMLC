# The proporial linear solver can be converted to a root finder by replacing one line.
# y1 = m*m instead of y1=m*x will allow m to converge to the square root of m

# Output, input and initial m
y = 5
#x = 2
m = 0.5

# Hyper parameters
# Learning rate
l = 0.25
n = 10

print("i,m,y1")

for i in range(0,n):
	# Predict the square
	y1 = m*m
	print(i,m,y1)

	# Update weight, the weight will converge to the square root of y
	m = m + l*(y-y1)
