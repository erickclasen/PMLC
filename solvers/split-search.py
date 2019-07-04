''' Simple Almost Machine Learning Linear Solver with Split Search. '''

# Output, input and initial m
y = 5
x = 2
m = 0

# Hyper parameter
# Learning rate.
l = 1.0

print("i,m,y1")

while l > 0.01:

        y1= m*x

        if y1 <= y:
                m += l
        elif y1 > y:
                l = l/2
                m -= l

        print(l,m,y1)
