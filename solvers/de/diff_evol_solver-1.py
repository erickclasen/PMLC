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

# Output, input and initial m
y = 5
x = 2
w = 0


# An estimate of teh bounds of the problem, the random numbers are generated within this range.
# Clipping of the pop values can be done to this range as well.
bounds = [0,5]

norm_bias = np.mean(bounds)+bounds[0]
norm_weight = bounds[1]-bounds[0]

# Mutation constant
mut = 1
pop_size = 10
dimensions = 1

# For each position, we decide (with some probability defined by crossp) if that number will be replaced or not by the one in the mutant at the same position.
crossp = 0.5

# The array of mutations, dimensions related to the number of variables or parameters.
mutant=[0 for i in range(dimensions)]

# Generate a starter random population normalized to the middle of the range
pop = (np.random.rand(pop_size, dimensions) * norm_weight) + norm_bias

print("Initial Population")
print(pop)
print()

# Outer loop of epochs of differential evolution.
for m in range(0,50):
        # Iterate through population.
        for i in range(0,pop_size):

                # Predict with the i-th population value.
                y1 = pop[i][0]*x 
                
                # Create indexes filled to the pop_size without the index that we are on.       
                idxs = [idx for idx in range(pop_size) if idx != i]

                # Randomly choose 3 indexes without replacement. 
                selected = np.random.choice(idxs, 3, replace=False)

                # Pick out the three that will be used for the trial mutation from the selection.
                a,b,c = pop[selected]

                # Generate the mutation to run a trial with.
                mutant = mut*a+(b - c) 
                # Make it fit into the bounds, if it is out after the operation above.
                np.clip(mutant, bounds[0], bounds[1])

                # binomial crossover since the number of selected locations follows a binomial distribution.
                cross_points = np.random.rand(dimensions) < crossp
                
                if not np.any(cross_points):
                        cross_points[np.random.randint(0, dimensions)] = True

                # Create a trial version to test out.
                trial = np.where(cross_points, mutant, pop[i])

                # Trial prediction using the mutation.  
                y2 = trial[0] * x 

                # Errors of the population at the idx and the trial mutation.
                e1 = abs(y - y1)
                e2 = abs(y - y2)

                # If the mutation is better, keep it and store it in the population.
                if e1 > e2:
                        # The mutation is better. Overwrite the population at index i.
                        pop[i] = trial
                        print(m,trial,y1)

# How good is it so far?
print("Index and y1 prediction:",m,y1)

