''' No brains, just brute force search solver. '''

# Output, input and initial m
y = 5
x = 2
m = 0

# Hyper parameters
# Learning rate and max number of loops.
l = 0.25
n = 20

print("i,m,y1")

for i in range(0,n):
	# Predict
	y1 = m*x
	print(i,m,y1)

	# Update weight
	m = m + l

        # Done solving, break out of the loop!
	if y1 ==y:
		break
