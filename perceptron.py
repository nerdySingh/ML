import numpy as np
from numpy import *
import matplotlib.pyplot as plt

weights = np.array([1,-1,0,0.5])
net = []
console = []
d = np.array([-1,-1,1])
x = np.array([[1,-2,0,-1],[0,1.5,-0.5,-1],[-1,1,0.5,-1]])
for i in np.arange(len(x[0])-1):  
    temp=0
    for j in np.arange(len(x[0])):
        temp = temp + (x[i][j]*weights[j])
    if d[i] >0 and temp < 0 or d[i]<0 and temp>0 :
        o = np.sign(temp)
        for j in np.arange(len(weights)):
            weights[j] = weights[j]+(0.1 *(d[i] -o)* x[i][j])
    net.append(temp)
    if net[i]>0:
        plt.plot(net[i],"r",marker="o")
    else:
        plt.plot(net[i],"b",marker="^")
plt.show()
print(net)