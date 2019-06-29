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


def NAND_percep(x):
    w = np.array([-1, -1])
    b = 1.5
    return perceptron(x, w, b)

# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("NAND({}, {}) = {}".format(1, 1, NAND_percep(example1)))
print("NAND({}, {}) = {}".format(1, 0, NAND_percep(example2)))
print("NAND({}, {}) = {}".format(0, 1, NAND_percep(example3)))
print("NAND({}, {}) = {}".format(0, 0, NAND_percep(example4)))


