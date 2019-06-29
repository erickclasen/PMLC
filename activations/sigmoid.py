import numpy as np
from matplotlib import pyplot as plt

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#-----------------------------------------

X = np.arange(-10,10,.1)
Y = sigmoid(X)
plt.plot(X,Y, c="blue")

# Format plot.
title = "Sigmoid Activation"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)

plt.ylabel("Activation Output", fontsize=12)
plt.show()

# The Sigmoid Derivative Plotted

X = np.arange(-10,10,.1)
Y = sigmoid_prime(X)
plt.plot(X,Y, c="red")

# Format plot.
title = "Sigmoid Derivative"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)
#fig.autofmt_xdate()
#plt.legend(loc='best')

plt.ylabel("Sigmoid Deriv. Output", fontsize=12)

plt.show()

