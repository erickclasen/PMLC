''' This code can load pretrained and saved weights and biases and classify the binary input to the one hot output vector.
    Serves as an example of saving and loading weights and biases from a file. '''
import numpy as np
import json

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


filename = 'weights.json'

# If the file exists read it in, if not just create it on the first pass through.
try:
        with open(filename) as f_obj:
                 w_list = json.load(f_obj)

except IOError:
        print("Missing File")
        quit()

filename = "biases.json"

# If the file exists read it in, if not just create it on the first pass through.
try:
        with open(filename) as f_obj:
                 b_list = json.load(f_obj)

except IOError:
        print("Missing File")
        quit()


weights = np.array(w_list)
biases = np.array(b_list)

#print(weights,biases)

print("Try out a value using the trained weights and biases.")

X = np.array([  [0,0,0,1] ] )
print(X)
print()
# forward propagation
l0 = X
l1 = nonlin(np.dot(l0,weights) + biases)
print(l1)
print("Clip all the low values to make it easier to read...")
lt = (l1 > 0.5) * l1
print(lt)

