import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 10, 10, 10

howmany = 10
batches = howmany/10
# Create random input and output data
#x = np.random.randn(N, D_in)
#y = np.random.randn(N, D_out)
xx = np.zeros(howmany)

yy = np.arange(0,howmany,1)

for n in range(0,howmany):
   	xx[n] = yy[n]

# Need the brackets around the vectors as this has to be 2d
x = np.array([xx])
y = np.array([yy])

# Make the long arrays into batches.
x = x.reshape(batches,10)
y = y.reshape(batches,10)


xa = np.array([[0,1,4,9,16,25,36,49,64,81]])
ya = np.array([[0,1,2,3,4,5,6,7,8,9]])


print(xa,ya)
print()
print(x,y)

#quit()

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# One hidden_layer bias and output bias
b1 = np.random.randn(1, H)
b2 = np.random.randn(1 , D_out)


learning_rate = 1e-6
for t in range(1000):
    # Forward pass: compute predicted y
    h = x.dot(w1) + b1
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2) + b2

    # Compute and print loss
    if t % 100 == 0:
	loss = np.square(y_pred - y).sum()
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
    #b1 -= learning_rate * np.sum(grad_w1.T,axis=0,keepdims=True) 
    #b2 -= learning_rate * np.sum(grad_w2.T,axis=0,keepdims=True) 


    # This is probably wrong!!!	
    b1 -= learning_rate * grad_y_pred
    b2 -= learning_rate * grad_h 


print(y_pred)
print(t)
#print(b1)
