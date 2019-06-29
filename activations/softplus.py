'''
https://jamesmccaffrey.wordpress.com/2017/06/23/two-ways-to-deal-with-the-derivative-of-the-relu-function/
A second alternative is, instead of using the actual y = ReLU(x) function, 
use an approximation to ReLU which is differentiable for all values of x.
 One such approximation is called softplus which is defined y = ln(1.0 + e^x) 
which has derivative of
 y' = 1.0 / (1.0 + e^-x) 
 
which is, remarkably, the logistic sigmoid function. Neat!
'''


import numpy as np
from matplotlib import pyplot as plt

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(z):
    z[z > 0] = 1
    z[z <= 0] = 0

    return z



def softplus(z,deriv=False):

    if deriv == False:
    	return np.log(1.0 + np.exp(z))
    else:
	return (1.0 / (1.0 +  np.exp(-z)))

#-----------------------------------------

X = np.arange(-10,10,.1)
Y = softplus(X)
plt.plot(X,Y, c="blue")


# Format plot.
title = "Softplus Activation"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)

plt.ylabel("Activation Output", fontsize=12)
plt.show()

# The Derivative Plotted

X = np.arange(-10,10,.1)
Y = softplus(X,True)
plt.plot(X,Y, c="red")



# Format plot.
title = "Softplus Derivative"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)
#fig.autofmt_xdate()
#plt.legend(loc='best')

plt.ylabel("Sigmoid Deriv. Output", fontsize=12)

plt.show()

