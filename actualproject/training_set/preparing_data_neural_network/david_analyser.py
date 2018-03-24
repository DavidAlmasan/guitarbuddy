import sys, getopt, re
import numpy as np
from matplotlib import pyplot as plt 

inputfile = "accuracy_vector.txt"
outputfile = open("out.txt", "w")
delimiters = "[", "]", " "
regexPattern = '|'.join(map(re.escape, delimiters))
with open(inputfile) as f:
    lines = f.readlines()
lines = [re.split(regexPattern, x) for x in lines] 
#lines = [[int(x[1]), float(x[2])] for x in lines]
x = np.arange(len(lines))
y = [float(lines[i][-2]) for i in range(len(lines))]
print(np.max(y))
for i in range(len(x)):
	if y[i] == np.max(y):
		print(i, ',', y[i])
		print("epoch", 800//15, "for max accuracy")
		break
plt.plot(x, y)
plt.ylim(0.95, 0.97)
plt.show()
