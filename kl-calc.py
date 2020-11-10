from pyomo.gdp import *
import numpy as np
import math
file = open('probs.txt', 'r')
Lines = file.readlines()
n = len(Lines)


p = [0]
q = [0]


for line in Lines:
    probs=line.split(",")
    q = np.concatenate((q, np.array([float(probs[0])])), axis=0)
    p = np.concatenate((p, np.array([float(probs[1])])), axis=0)

q = np.delete(q, [0])
p = np.delete(p, [0])

sum=0
for j in range(n):
    sum+=p[j]*math.log(p[j]/q[j])
print(sum)