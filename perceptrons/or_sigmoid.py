import numpy as np

''' Shows how for a large value of C the sigmoid function approaches a unit step function behavior.
    http://neuralnetworksanddeeplearning.com/chap1.html Sigmoid neurons simulating perceptrons, part II  '''

def unit_step(v):
        """ Heavyside Step function. v must be a scalar """
        if v >= 0:
                return 1
        else:
                return 0

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

        
def perceptron(x, w, b):
#     Function implemented by a perceptron with 
#               weight vector w and bias b """
        v = np.dot(w, x) + b
        #y = unit_step(v)
        # Use the Sigmoid non-liniarity, which for a large C gives the same result as the unit step function.
        y = nonlin(v)
        return y


def OR_percep(x):
    w = np.array([1, 1])
    b = -0.5
    C = 1000

    # Multiply the weights and biases by C, as C approaches in the limit, infinity the sigmoid approaches infinity.     
    w = w * C
    b = b * C
    #print(w*x+b)
    return perceptron(x, w, b)

# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])
example5 = np.array([-500, 1])

print("OR({}, {}) = {}".format(1, 1, OR_percep(example1)))
print("OR({}, {}) = {}".format(1, 0, OR_percep(example2)))
print("OR({}, {}) = {}".format(0, 1, OR_percep(example3)))
print("OR({}, {}) = {}".format(0, 0, OR_percep(example4)))

print("Force w * x + b to zero")
print("OR({}, {}) = {}".format(1, 1, OR_percep(example5)))


