from matplotlib import pyplot as plt
import numpy

f = open("graph.txt", 'r')
data = [x.split() for x in f.readlines()]
x = [float(x[0]) for x in data]
y = [float(x[1]) for x in data]


plt.plot(x, y)
plt.show()

