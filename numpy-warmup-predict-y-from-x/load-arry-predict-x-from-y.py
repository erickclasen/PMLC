import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 10, 10, 10

# Create random input and output data
#x = np.random.randn(N, D_in)
#x = np.array([[0,1,4,9,16,25,36,49,64,81],[1,4,9,16,25,36,49,64,81,100]])
#y = np.random.randn(N, D_out)
#y = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10]])
#x = np.array([[0,1,4,9,16,25,36,49,64,81]])
y = np.array([0,1,2,3,4,5,6,7,8,9])
x = np.array(np.loadtxt('test.txt', ndmin=1))
#x = np.squeeze(np.asarray(x))
#print("---->",x)
#x = np.ones((1,10))
#y = np.ones((1,10))
#x= np.fromfile('test.txt'(1,10))
#for countp in (0,10,1):
#   result = countp # do_stuff(line)
#    x = np.append(x, [result], axis=1)
#    print countp
print("X and Y numpy arrays")
print(x,y)
print"--------------"
# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(250000):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if(t % 10000 == 0):
	print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
print("")
print("Results X")
print(x)
print("Y  ---------------")
print(y)
print("Predicated Y ----------------")
print(y_pred)
#print("--------Weights---------")
#print("w1",w1)
#print("w2",w2)
print("-----------------------")
print("Load x with new array and make a prediction with model weights set")
#x = np.array([[0,1,9,16,20,25,36,50,81,100]])
x = np.array([[0,1,3,8,15,24,35,48,64,81]])
x = np.array([[0,1,4,9,16,25,36,49,64,80]])
x = np.array([[1,4,9,16,25,36,49,64,81,100]])

h = x.dot(w1)
h_relu = np.maximum(h, 0)
yy_pred = h_relu.dot(w2)
print(x,yy_pred)
