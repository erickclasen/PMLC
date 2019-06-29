import numpy as np
from matplotlib import pyplot as plt
import copy

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
    # Do a copy to avoid the pointer issue.
    d = copy.deepcopy(z)

    d[d > 0] = 1

    # Make the deriv 0 at 0 as it is undefined, this preserves continuity.
    d[d <= 0] = 0

    return d


#-----------------------------------------

X = np.arange(-10,10,.1)
Y = relu(X)
plt.plot(X,Y, c="blue")


# Format plot.
title = "Relu Activation"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)

plt.ylabel("Activation Output", fontsize=12)
plt.show()

# The Sigmoid Derivative Plotted

X = np.arange(-10,10,.1)
Y = relu_deriv(X)
#print(X,Y)
plt.plot(X,Y, c="red")



# Format plot.
title = "Relu Derivative"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)

plt.ylabel("Relu Deriv. Output", fontsize=12)

plt.show()

