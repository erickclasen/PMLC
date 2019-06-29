import math 

learnstep = 1.0
value = 0.0
y_max = 0
y_last = 0

for n in range (1,40,1):
	y= math.sin(n/10.0)
	if y > y_last:
		y_last = y
	print(n,y,y_last)


while learnstep > 0.1:

	y= math.sin(value/10.0)

	if y >= y_max:
		y_max = y
		value += learnstep
	else:
		#continue
		learnstep = learnstep/2
                value -= learnstep

	print(value,y,y_max,learnstep)
