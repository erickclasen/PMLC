import numpy as np

x = np.arange(8)
x_as_array = np.array(x)
print(x,x_as_array)


print("Shape to 2,4")
array = x.reshape(2,4)
print(array)


print("Transposed to 4,2")
array = x.reshape(4,2)
print(array)


print("3D Array")
array = x.reshape(2,2,2)
print(array)


print("Transpose of the original array,")
array = x.reshape(8,1)
print(array)
print(array.T)

print("Transpose of the original x made into array, stays horizontal")
print(x_as_array.T)
print("Notice this does not work either!")
print(x.T)

