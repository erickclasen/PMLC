import numpy as np
print("Not sure why it doesn't work.")
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Output, input and initial m
y = 5
x = 2
m = 0.5

#x = nonlin(x)
#y = nonlin(y)
# Hyper parameters
# Learning rate
l = 0.025
n = 100

print("i,m,y1")

for i in range(0,n):
	# Predict
	y1 = nonlin(m*x)
	print(i,m,y1)

	# Update weight
	m = m + l*(y-nonlin(y1,True))


# logit function, the inverse of sigmoid x = ln(y/(1-y))
final = np.log(y1/(1-y1))
print(final)
print("Why 2?")
