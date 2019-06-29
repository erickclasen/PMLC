# Output, input and initial m
#y = 5
x = 2
m = 0.5

# Hyper parameters
# Learning rate
l = 0.25
n = 10

print("i,m,y1")

for i in range(0,n):
	# Predict
	y1 = x/m
	print(i,m,y1)

        # Recursive with respect to m.
	m = (m + y1)/2
