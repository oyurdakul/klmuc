from subproblem import *
from masterproblem import *
from readwrite import *
import time as tm
import numpy as np
import os
st=tm.time()
tt=24; tps=1; al_x1=np.zeros(tps); al_la=0; al_mu=0; be=-50000; \
    ub=100000; lb=-50000; max_hw=180000;  scaling=1;\
    gcostc1=0.018/scaling; glim1=50/scaling; gcostl1=0.179; pramp1=30/scaling; nramp1=-20/scaling;\
    stc1 = 1/scaling; mut1 = 1; mdt1 = 1;\
x1 = [0]
objs=[0]
snns=8
gen1 = np.zeros((1, snns))
sellls = np.zeros((1, snns))
buyys = np.zeros((1, snns))
for t in range(tt):
    print("current t value:", t)
    al_x1 = np.zeros(tps);
    al_la = 0;
    al_mu = 0;
    be = -50000; \
    ub = 100000;
    lb = -50000;
    rho = 0.4;


    max_hw = 180000;
    scomp = 30;
    i = 1

    seninp = 'sces/sc'+str(t+1)+'.txt';
    while (abs(ub-lb)>0.00001):
        rw(t, i,al_x1, al_la, al_mu, be, max_hw, scomp, gcostc1, rho, tps, stc1, mut1, mdt1,)
        f_x1, f_la, f_mu, lb, obj = mpr(t)
        maxim, ub, sn, gens1, buys, sells, al_x1, al_la, al_mu, be, netload = subp(seninp, \
            scaling, f_x1, f_la, f_mu, tps, gcostl1, glim1, pramp1, \
            nramp1, gcostc1, )
        if i==1:
            max_hw=maxim
        i += 1
    x1 = np.concatenate((x1, np.array(f_x1)), axis=0)
    objs = np.concatenate((objs, np.array([obj])), axis=0)
    gen1 = np.concatenate((gen1, np.array(gens1)), axis=0)
    sellls = np.concatenate((sellls, np.array(sells)), axis=0)
    buyys = np.concatenate((buyys, np.array(buys)), axis=0)

x1 = np.delete(x1, 0, axis=0)

objs = np.delete(objs, 0, axis=0)

gen1 = np.delete(gen1, 0, axis=0)

sellls = np.delete(sellls, 0, axis=0)

buyys = np.delete(buyys, 0, axis=0)

for i in range(tt):
    print("objective in time period ", i + 1, objs[i]*scaling)
print("total objective:", sum(objs)*scaling)
for i in range(tt):
    for j in range(sn):
        print("generator 1 generation in time period", i+1, "scenario", j+1, gen1[i,j]*scaling)

for i in range(tt):
    for j in range(sn):
        print("purchased power in time period", i+1, "scenario", j+1, buyys[i,j]*scaling)

for i in range(tt):
    for j in range(sn):
        print("curtailed power in time period", i+1, "scenario", j+1, sellls[i,j]*scaling)

for i in range(tt):
    print("x1 commitment in time period ",i+1, x1[i])
os.system('afplay /System/Library/Sounds/Sosumi.aiff')
# et=tm.time()
# print("duration: ", et-st)
#
# gensum=np.zeros(sn)
# for i in range(sn):
#     temp=0
#     for j in range(tps):
#         temp  += (gens1[i,j] + gens2[i,j] + gens3[i,j] + buys[i,j] - sells[i,j])
#     gensum[i]=temp
# nl=np.zeros(sn)
# for i in range(sn):
#     temp=0
#     for j in range(tps):
#         temp  += netload[i,j]
#     nl[i]=temp
#
# f = open("results_rho08.txt", "w")
# f.write("duration:")
# f.write(str(et-st))
# f.write(("\n"))
# f.write("scomp:")
# f.write(str(scomp))
# f.write(("\n"))
# f.write("upper bound:")
# f.write(str(ub))
# f.write(("\n"))
# f.write("lower bound:")
# f.write(str(lb))
# f.write(("\n"))
# f.write("obj funct value:")
# f.write(str(obj))
# f.write(("\n"))
# f.write("first stage decision variable x1:")
# f.write(str(f_x1))
# f.write(("\n"))
# f.write("first stage decision variable x2:")
# f.write(str(f_x2))
# f.write(("\n"))
# f.write("first stage decision variable x3:")
# f.write(str(f_x3))
# f.write(("\n"))
# f.write("first stage decision variable lambda:")
# f.write(str(f_la))
# f.write(("\n"))
# f.write("first stage decision variable mu:")
# f.write(str(f_mu))
# f.write(("\n"))
# for i in range(sn):
#     for j in range(tps):
#         f.write("First generator generation in time period ")
#         f.write(str(j+1))
#         f.write(", scenario ")
#         f.write(str(i+1))
#         f.write(": ")
#         f.write(str(gens1[i,j]))
#         f.write("\n")
# for i in range(sn):
#     for j in range(tps):
#         f.write("Second generator generation in time period ")
#         f.write(str(j+1))
#         f.write(", scenario ")
#         f.write(str(i+1))
#         f.write(": ")
#         f.write(str(gens2[i,j]))
#         f.write("\n")
# for i in range(sn):
#     for j in range(tps):
#         f.write("Third generator generation in time period ")
#         f.write(str(j+1))
#         f.write(", scenario ")
#         f.write(str(i+1))
#         f.write(": ")
#         f.write(str(gens3[i,j]))
#         f.write("\n")
# for i in range(sn):
#     for j in range(tps):
#         f.write("Purchased power in time period ")
#         f.write(str(j+1))
#         f.write(", scenario ")
#         f.write(str(i+1))
#         f.write(": ")
#         f.write(str(buys[i,j]))
#         f.write("\n")
# for i in range(sn):
#     for j in range(tps):
#         f.write("Sold power in time period ")
#         f.write(str(j+1))
#         f.write(", scenario ")
#         f.write(str(i+1))
#         f.write(": ")
#         f.write(str(sells[i,j]))
#         f.write("\n")
# for i in range(sn):
#     f.write("Net generated power in scenario ")
#     f.write(str(i+1))
#     f.write(": ")
#     f.write(str(gensum[i]))
#     f.write(", net load in scenario ")
#     f.write(str(i+1))
#     f.write(": ")
#     f.write(str(nl[i]))
#     f.write("\n")
# f.close()
#
# print("upper bound:", ub)
# print("lower bound:", lb)
# print(ub-lb)
# print("objective function value:", obj)
# # print("number of scenarios: ", sn)
# print("first stage decision variable x1", f_x1)
# print("first stage decision variable x2", f_x2)
# print("first stage decision variable x3", f_x3)
# print("first stage decision variable lambda", f_la)
# print("first stage decision variable mu", f_mu)
# #
#
#
# for i in range(sn):
#     for j in range(tps):
#         print("First generator generation in time period ", j+1, ", scenario ", i+1,": ",\
#               gens1[i,j])
#
# for i in range(sn):
#     for j in range(tps):
#         print("Second generator generation in time period ", j+1, ", scenario ", i+1,": ",\
#               gens2[i,j])
# for i in range(sn):
#     for j in range(tps):
#         print("Purchased  power in time period ", j+1, ", scenario ", i+1,": ", buys[i,j])
#
# for i in range(sn):
#     for j in range(tps):
#         print("Sold  power in time period ", j+1, ", scenario ", i+1,": ", sells[i,j])
