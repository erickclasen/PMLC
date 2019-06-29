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

def NOT_percep(x):
        return perceptron(x, w=-1, b=0.5)

def AND_percep(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptron(x, w, b)

def NAND_percep(x):
    w = np.array([-1, -1])
    b = 1.5
    return perceptron(x, w, b)


def OR_percep(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)

def XOR_net(x):
    gate_1 = NAND_percep(x)
    combine0 = np.array( [x[0],gate_1])

    gate_2 = NAND_percep(combine0)
    combine1 = np.array( [x[1],gate_1])

    gate_3 = NAND_percep(combine1)
    combine3 = np.array([gate_2, gate_3])

    output = NAND_percep(combine3)
    return output


# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("XOR({}, {}) = {}".format(1, 1, XOR_net(example1)))
print("XOR({}, {}) = {}".format(1, 0, XOR_net(example2)))
print("XOR({}, {}) = {}".format(0, 1, XOR_net(example3)))
print("XOR({}, {}) = {}".format(0, 0, XOR_net(example4)))
