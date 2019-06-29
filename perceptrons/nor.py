import numpy as np

def unit_step(v):
	""" Heavyside Step function. v must be a scalar """
	if v >= 0:
		return 1
	else:
		return 0
	
def perceptron(x, w, b):
#     Function implemented by a perceptron with 
#		weight vector w and bias b """
	v = np.dot(w, x) + b
	y = unit_step(v)
	return y
def OR_percep(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)


def NOR_percep(x):
    w = np.array([-1, -1])
    b = 0.5
    return perceptron(x, w, b)

# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("NOR({}, {}) = {}".format(1, 1, NOR_percep(example1)))
print("NOR({}, {}) = {}".format(1, 0, NOR_percep(example2)))
print("NOR({}, {}) = {}".format(0, 1, NOR_percep(example3)))
print("NOR({}, {}) = {}".format(0, 0, NOR_percep(example4)))


