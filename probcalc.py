from pyomo.gdp import *
from pyomo.environ import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory
import numpy as np
import math
file = open('probs.txt', 'r')
Lines = file.readlines()
n = len(Lines)
model=ConcreteModel()
model.s1=RangeSet(n)
model.p=Var(model.s1, within=PositiveReals)
model.q=Param(model.s1, mutable=True)
rho=0.03

j=1
for line in Lines:
    probs=line.split(",")
    model.q[j] = float(probs[0])
    j+=1

def kl_div_const(model):
    sum=0
    for j in model.s1:
        sum += model.p[j] * log(model.p[j] / model.q[j])
    return sum<=rho

def prob_const(model):
    sum = 0
    for j in model.s1:
        sum += model.p[j]
    return sum==1

def prob_const_2(model, j):
    return model.p[j]<=1

def obj(model):
    sum = 0
    for j in model.s1:
        sum += model.p[j] * log(model.p[j] / model.q[j])
    return sum

model.c1=Constraint(rule=kl_div_const)
model.c2=Constraint(rule=prob_const)
model.c3=Constraint(model.s1, rule=prob_const_2)

model.obj=Objective(sense=maximize, rule=obj)
opt = SolverFactory('baron')
opt.solve(model)
model.p.display()
