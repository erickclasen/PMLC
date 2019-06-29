import math 

learnstep = 1.0
value = 0.0
y_last = 0

for n in range (1,40,1):
	y= math.sin(n/10.0)
	if y > y_last:
		y_last = y
	print(n,y,y_last)

last_y = 0.0
while learnstep > 0.1:

	y= math.sin(value/10.0)

	if y >= last_y:
		value += learnstep
	elif y < last_y:
		learnstep = learnstep/2
                value -= learnstep
	y_last = y

	print(value,y,last_y,learnstep)
