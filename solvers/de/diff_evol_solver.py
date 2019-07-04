''' Differential Evolution - Create a randowm population which is mean zero and dtd dev of 1, as in a normal distribution.
Take it and scale and shift it via a weight and bias to the range of the bounds, therefore normalizing it.
Copy the population to popcp and select from three random places in the population. Take these values and form a new
trial mutation value by adding and differencing, trial = a * mut + (b-c)
 Try this value out in the y = m * x formula. Also try the value pulled from the population, which is
looped through in the inner loop. If the mutation trial does better save it at the location that is at the array position that
is currently being looped through. Repeat until the outer loop ends.
Hyper parameters.
bounds - A range within the value is expected to fall.
mut - Mutation constant. From 0.5-2.0. Sets how aggressive the mutation process is per iteration.
pop_size - The population size.
'''

import numpy as np

# Trial is an array of 3 to create the mutation from.
trial = [0,0,0]

# Output, input 
y = 5
x = 2

# An estimate of teh bounds of the problem, the random numbers are generated within this range.
# Clipping of the pop values can be done to this range as well.
bounds = [0,5]

norm_bias = np.mean(bounds)+bounds[0]
norm_weight = bounds[1]-bounds[0]
#print(norm_weight,norm_bias)

# Mutation constant
mut = 1
pop_size = 10
#dimensions = 1

# Generate a starter random population normalized to the middle of the range
pop = (np.random.rand(pop_size, 1) * norm_weight) + norm_bias

print("Initial Population")
print(pop)
print()


# Outer loop of epochs of differential evolution.
for m in range(0,50):
	# Iterate through population.
	for i in range(0,pop_size):

		# Make a copy to work with and remove values used
		popcp = pop[:]
		# Predict with the i-th population value.
		y1 = pop[i]*x
		# Remove it from the copy.
		popcp = np.delete(popcp,i,0)

		for s in range(0,3):
			#print(s,len(popcp),popcp)
			select = np.random.randint(0,len(popcp))
			#print(select)
			# Remove the trial from the list at the select point.
			trial[s] = popcp[select]
			popcp = np.delete(popcp,select,0)

		# These are the trials to mutate.
		#print(trial[0],trial[1],trial[2])

		# Generate the mutation to run a trial with.
		muta = mut*trial[0]+(trial[1] - trial[2]) 

		# Trial prediction using the mutation.	
		y2 = muta * x

		# Errors of the population at the idx and the trial mutation.
		e1 = abs(y - y1)
		e2 = abs(y - y2)

		# If the mutation is better, keep it and store it in the population.
		if e1 > e2:
			pop[i] = muta
			print(m,muta)
	# How good is it so far?
print("Index and y1 prediction:",m,y1)

