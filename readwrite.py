from subproblem import *
from masterproblem import *
import os
#Updated read-write file
def rw(t, i, al_x1, al_la, al_mu, be, max_hw, scomp, gcostc1, rho, tps, \
       stc1, mut1, mdt1,):
    c="alpha_x1_"+str(t)+".txt"
    f = open(c, "a")
    for j in range(1, tps+1):
        f.write(str(i))
        f.write(" ")
        f.write(str(j))
        f.write(" ")
        f.write(str(al_x1[j-1]))
        f.write("\n")
    f.close()


    c = "alpha_la_" + str(t) + ".txt"
    f = open(c, "a")
    f.write(str(i))
    f.write(" ")
    f.write(str(al_la)+"\n")
    f.close()
    c = "alpha_mu_" + str(t) + ".txt"
    f = open(c, "a")
    f.write(str(i))
    f.write(" ")
    f.write(str(al_mu)+"\n")
    f.close()
    c = "betas_" + str(t) + ".txt"
    f = open(c, "a")
    f.write(str(i))
    f.write(" ")
    f.write(str(be))
    f.write("\n")
    f.close()
    data = "param ccount:="
    data += str(i)
    data += ";\n"
    data += "param max_hw:="
    data += str(max_hw)
    data += ";\n"
    #First stage costs are now passed from the main file
    data += "param gcostc1:="
    data += str(gcostc1)
    data += ";\n"

    #Rho is now passed from the main file
    data += "param rho:="
    data += str(rho)
    data += ";\n"
    data += "param scomp:="
    data += str(scomp)
    data += ";\n"
    data += "param pn:="
    data += str(tps)
    data += ";\n"
    data += "param stc1:="
    data += str(stc1)
    data += ";\n"
    data += "param mut1:="
    data += str(mut1)
    data += ";\n"
    data += "param mdt1:="
    data += str(mdt1)
    data += ";\n"

    data11 =  data2 = data3 = data4 = ""
    c = "alpha_x1_" + str(t) + ".txt"
    with open(c) as fp:
        data11 = fp.read()

    c = "alpha_la_" + str(t) + ".txt"
    with open(c) as fp:
        data2 = fp.read()
    c = "alpha_mu_" + str(t) + ".txt"
    with open(c) as fp:
        data3 = fp.read()
    c = "betas_" + str(t) + ".txt"
    with open(c) as fp:
        data4 = fp.read()

    data += "param alpha_x1:=\n"
    data += data11
    data += ";\n\n"

    data += "param alpha_la:=\n"
    data += data2
    data += ";\n\n"
    data += "param alpha_mu:=\n"
    data += data3
    data += ";\n\n"
    data += "param beta:=\n"
    data += data4
    data += ";\n"
    c = "master1_" + str(t) + ".dat"
    with open(c, 'w') as fp:
        fp.write(data)