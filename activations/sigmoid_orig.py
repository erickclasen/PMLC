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
for z in range(-100,100,1):
	s = sigmoid(z/10.)
        print(z/10.,s)
	plt.plot(z/10.,s)
    	plt.scatter(z/10.,s, c="blue",edgecolor='none', s=5)




# Format plot.
title = "Sigmoid Activation"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)

plt.ylabel("Activation Output", fontsize=12)
plt.show()

# The Sigmoid Derivative Plotted
for z in range(-100,100,1):
        s = sigmoid_prime(z/10.)
        print(z/10.,s)
        plt.plot(z/10.,s)
        plt.scatter(z/10.,s, c="red",edgecolor='none', s=5)

# Format plot.
title = "Sigmoid Derivative"
plt.title(title, fontsize=16)
plt.xlabel('', fontsize=10)
#fig.autofmt_xdate()
#plt.legend(loc='best')

plt.ylabel("Sigmoid Deriv. Output", fontsize=12)

plt.show()

