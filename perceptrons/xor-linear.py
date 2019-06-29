import numpy as np

# 

def unit_step(v):
        """ Heavyside Step function. v must be a scalar """
        if v >= 0:
                return 1
        else:
                return 0

def perceptron(x, w, b):
#     Function implemented by a perceptron with 
#               weight vector w and bias b """
        v = np.dot(w, x) + b
        y = unit_step(v)
        return y

def XOR_percep(x):
    #  Feature engineering, multiply the x inputs together and append to array.    
    x = np.append(x,x[0]*x[1])
    # Add a new weight in that works with the multipled x values in the array.
    w = np.array([1, 1, -2])
    b = -0.5
    return perceptron(x, w, b)

# Left in here just for comparision
def XOR_net(x):
    gate_1 = AND_percep(x)
    gate_2 = NOT_percep(gate_1)
    gate_3 = OR_percep(x)
    new_x = np.array([gate_2, gate_3])
    output = AND_percep(new_x)
    return output


# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("XOR({}, {}) = {}".format(1, 1, XOR_percep(example1)))
print("XOR({}, {}) = {}".format(1, 0, XOR_percep(example2)))
print("XOR({}, {}) = {}".format(0, 1, XOR_percep(example3)))
print("XOR({}, {}) = {}".format(0, 0, XOR_percep(example4)))
