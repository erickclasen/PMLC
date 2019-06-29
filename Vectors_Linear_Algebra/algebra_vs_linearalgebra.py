import numpy as np

# Multiply two arrays 
x = [1,2,3]
y = [2,3,4]
product = []

print("x,y",x,y)

print("i,x[i],y[i],product")
for i in range(len(x)):
    product.append(x[i]*y[i])
    print(i,x[i],y[i],product)	
print("Algebra:",product)

    
# Linear algebra version (3x faster)
x = np.array([1,2,3])
y = np.array([2,3,4])
product = x * y

print("Linear Algebra:",product)

dot_prod = np.dot(x,y)
print("Dot Prod",dot_prod)


# input dataset
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1] ])

# output dataset
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
print("X",X)
print("y",y)
print("syn0",syn0)
print("")

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
print("inputs",inputs)
print("expected_output",expected_output)
print("hidden_weights",hidden_weights)
