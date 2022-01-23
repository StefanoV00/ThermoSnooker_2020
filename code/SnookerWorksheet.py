# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:24:29 2020

@author: Stefano
"""

from ball_module import * # not necessary but for completeness
import simulation_module as Mod
from simulation_module import *
from func_module import *

import copy
import pylab as pl
import numpy as np
import sympy
from sympy.abc import pi


"""
GENERAL NOTES
"""

"""
1
THE MAXIMUM NUMBER OF BALLS FOR A CONTAINER WITH RADIUS Rc IS A BIT LESS THAN
    
    # Nmax <  pi / 4 / 1.5 **2 * (Rc/Rb) ** 2 
    # Nmax approx.=  0.31 * (Rc/Rb) ** 2

For Rb = 0.001, the default one, it means:

    # Nmax approx = 310,000 * Rc **2
    
ANYWAYS, IF A TOO BIG NUMBER IS GIVEN, THE SIMULATION CODE IS DESIGNED TO ACT
AS IF THE GIVEN N WAS EQUAL to Nmax, AND A NOTE WILL BE PRINTED, 
SO DON'T WORRY TO GET IT WRONG. HOWEVER, THAT DOES MAKE A DIFFERENCE IF
YOU HAVE MULTIPLE SPECIES OF BALLS: THEN self._Nj IS UNUSABLE.
"""


"""
2
THE RUN METHOD PRINTS A STATEMENT WHEN HALF THE FRAMES HAVE BEEN RUN:
    
     "We are at half the running time of this simulation"

IN THIS WAY, THE RUNNING PORCESS CAN BE TRACKED. 
"""

"""
3
IMPORTANT: if wanting to carry out analysis, the only thing that requires to
be changed are the initial paramters (lists or arrays). Very few commands are
in fact hardcoded (this implies many loops).
"""
   
"""
4
THE PLOTS HAVE BEEN SAVED MANUALLY: pl.savefig() WASN'T USED: 
I WANTED TO VISUALIZE THEM BEFORE SAVING THEM
"""

"""
5

#NOTE ON THE P-VALUES IN THE PLOTS: THEY ARE ROUNDED TO 5 DIGITS, HENCE, IF 
# THEY APPEAR BEING ZERO, IT MEANS THEY ARE SMALLER THAN 0.000001
"""


#%%

"Testing and Graphical Overview A"

#GIVES A GRAPHIC IDEA OF HOW THE SIMULATION WORKS
#therefore a not-so-big number of balls was chosen nor frames was chosen

sim_few = Mod.Simulation( R_container = 0.01 , R_balls = 0.001, N = 10)
print(sim_few.__repr__())

sim_few.run(100, True, 0.1)



#%%

"Testing and Graphical Overview B"

N = [1, 1, 2, 2]
m = np.array([1, 1, 4, 4]) * 1e-3
Rb = np.array([1, 4, 1, 4]) * 2e-3
c = ["black", "green", "blue", "red"]

sim_few = Mod.Simulation(m = m, R_container = 0.1 ,R_balls = Rb, N = N, c = c)

sim_few.R_set(0.15)
sim_few.plot()

sim_few.R_increase(1.5)
sim_few.plot()



#%%
"Testing and Graphical Overview C"

sim_few.run(50, True, 0.01)

# Prints the total pressure and the sum of the partial pressure
print(sim_few._P, sum(sim_few._Pj))

 
sim_few = Mod.Simulation(m = m, R_container = 0.1 ,R_balls = Rb, N = N, c = c)

# AS EXPLAINED IN THE ADD_BALLS METHOD DOC STRING, THE LABELS OF THE PLOTS
# WON'T BE UPDATED, AND SOME FEATURES ARE STILL TO BE PERFECTIONISED ABOUT
# THIS METHOD

sim_few.add_balls(N = 1, m = 0.002, Rb = 0.005, color = "yellow")
sim_few.plot()

sim_few.add_balls([1,2], [0.005, 0.0025], [0.001, 0.0005], ["orange", "cyan"])
sim_few.plot()



#%%

"""TASK 9,11a,13, pt1: 
    Momentum and Kinetic Energy and Angular Momentum graphs, 
    Position Histograms,
    Velocity Distributions 
    with no asymmetric boost"""
   
# IN ORDER TO PERFORM A STATISTICALLY SIGNIFICANT ANALYSIS, A BIG NUMBER 
# OF BALLS AND FRAMES WAS SELECTED TO BE IMPLEMENTED. 
# THE GRAPHICAL OUTPUT OF THE CELL IS:
    # THE GRAPHS OF MOMENTUM AND KINETIC ENERGY AND ANG. MOMENTUM IN TIME
    # A PLOT OF THE FINAL STATE OF THE SIMULATION
    # THE REQUESTED HISTOGRAMS COMPARED TO THE  THEORETICAL CURVES
    # The REQUESTED VELOCITY / SPEED DISTRIBUTIONS AND THEORETICAL CURVES

sim1 = Mod.Simulation( R_container = 0.1, N = 200)

sim1.run(50, plot_p = True, plot_K = True, plot_L = True)  

sim1.plot()   # the simulation in last frame

sim1.pos_hist()                

sim1.inter_pos_hist() 

sim1.v_distribution() 



#%%
"""TASK 9,11a,13, pt2:  
    Momentum graphs, 
    Position Histograms,
    Velocity Distributions 
    with asymmetric boost in the x direction"""
   
# SAME AS BEFORE, WITH A BOOST TO ALL PARTICLES GIVEN IN THE X DIRECTION

sim2 = Mod.Simulation( R_container = 0.025 , N = 100, xboost = 1.5)

sim2.run(200, plot_p = True)   

# sim2.plot()   # the simulation in last frame

# sim2.pos_hist()                

# sim2.inter_pos_hist() 

# sim2.v_distribution()



#%%

"""TASK 9,11a,13, pt3: multiple species
    Position Histograms, 
    Velocity Distributions 
    with no asymmetric boost"""
    
# SAME AS TWO CELLS BEFORE, BUT WITH MULTIPLE SPECIES NAD NO BOOST
    
N = [50, 50, 50, 50]
m = np.array([1, 1, 20, 20]) * 1e-3
Rb = np.array([1, 5, 1, 5]) * 1e-3
c = ["yellow", "red", "cyan", "blue"]

sim1m = Mod.Simulation( R_container = 1, N = N, m = m, R_balls = Rb, c = c )

sim1m.run(200)  
#%%
sim1m.plot()   

sim1m.pos_hist(multi = False)  # just for testing purposes
sim1m.pos_hist(multi = True) 
        
sim1m.v_distribution(multi = False) # just for testing purposes
sim1m.v_distribution(multi = True)



#%%
"Extra Task: Dalton's Law of Partial Pressures"

N = [10, 5, 5, 10]
m = np.array([1, 1, 20, 20]) * 1e-3
Rb = np.array([1, 5, 1, 5]) * 1e-3
c = ["orange", "red", "cyan", "blue"]

Rc = np.array([500, 600, 750, 900, 1000, ]) * 1e-3
S_DL = Rc ** 2
V_DL = np.pi * Rc ** 2
SimDL = []
P_DL = [] 
pP_DL = []
sP_DL = [[] for i in range(len(Rc))]

for i in range(len(Rc)):
    SimDL.append(Mod.Simulation(m, 0, Rc[i], Rb, N, c))
    SimDL[i].run(200)
    P_DL.append(SimDL[i].pressure())
    pP_DL.append(SimDL[i].pressures())

ratios = []
cumulative_ratios = [[] for x in range(len(Rc))]
cr = cumulative_ratios 

for i in range(len(Rc)):
    ratios.append(pP_DL[i] / P_DL[i])
ratios = np.array(ratios)

for i in range(len(Rc)):
    for j in range(len(N)):
        cr[i].append(sum(ratios[i][:(j+1)]))
cr = np.array(cr)
    
delta = S_DL[1:] - S_DL[:-1]
w = 0.9 * min(delta)

pl.figure()  
pl.axes(xlim = (0, 3.2 * max(S_DL))) 
pl.plot(S_DL, [1] * len(S_DL), "x", linewidth = 10, \
        label = "Total Normalized Pressures", color = "red")
 
for j in (range(1, len(N) + 1)):   
        pl.bar(S_DL, cr[:,-j], color = c[-j], width = w,\
               label = "Partial/Total Pressure Ratio\n \
m = %g kg, Rb = %g m, N = %g" % (m[-j], Rb[-j], N[-j]))
         
pl.legend(loc = "upper right")
pl.title("Total & Partial Pressures (Normalized) Relation" , size = 17, \
         family = "Times New Roman")
pl.xlabel("Volume (" + sympy.pretty(pi) + " m$^2$)", size = 13)
pl.ylabel("Pressure ratio", size = 13)
pl.grid()
pl.show()   



#%%
"""
THe following four tasks carry out the analysis of the idealness of a gas. The
capital letters following the quantities indicate:
    the first one the x-variable (the y is always pressure)
    the second one, if there is, what else varies
For instance RcTV is the list of radii of the container for the plot of 
pressure(Temperature) for different volumes
"""


"TASK 11b.1: Pressure vs Temperature for different volumes"

# FOR DIFFERENT VALUES OF THE RADIUS OF THE CONTAINER, ANLAYSES RELATION 
# BETWEEN KINETIC ENERGY OF THE PARTICLES (THE TEMPERATURE) AND PRESSURE
F = 50
SimTV = []
RcTV = np.array([25, 50]) * 1e-3
nTV = 10
boostTV = [1.5, 2, 3]

#1. Create simulation for the different values of Rc and velocity 
#   (hence temperature)
for i in range(len(RcTV)):
    simTV = Mod.Simulation(R_container = RcTV[i], N = nTV)
    SimTV.append(simTV)
    for j in range(len(boostTV)):
        SimTV.append(simTV.vel_increase(boostTV[j]))

T_TV = []
P_TV = []

#2.Find temperature and pressure for each
for i in range(len(SimTV)):
    SimTV[i].run(F)
    print("TV", i)
    T_TV.append(SimTV[i].temperature())
    P_TV.append(SimTV[i].pressure())
  

#3.Plot for Ideal Gas   
pl.figure(figsize = [8, 6])
pl.axes(ylim = [0, 1.1 * max(P_TV)])


for j in range(len(RcTV)):
    Ii = j * (len(boostTV)+1)
    If = (j+1) * (len(boostTV)+1)
    Tj = T_TV[Ii : If]
    Pj = P_TV[Ii : If]
    S =  RcTV[j] ** 2
    V =  np.pi * S
    
    Sstring = "{:.2e}".format(S)
    
    data = pl.plot(Tj, Pj, "x", label = "Empirical Data (V = " + Sstring \
                   + " " + sympy.pretty(pi) + " m$^2$)")
    c = data[0].get_color()
    X = np.linspace(0, 1.1 * max(Tj), 1000)
    gr = nTV * Mod.Simulation.kb / V 
    pl.plot(X, gr * X, ls = "dashed", color = c, label = "Ideal Gas Curve")


pl.legend(fontsize = 10)
pl.title("Pressure - Temperature Relation (N = %g)" % (nTV), size = 20, \
         family = "Times New Roman")
pl.xlabel("Temperature (K)", size = 15)
pl.ylabel("Pressure (Pa * m)", size = 15)
pl.grid()
pl.show()




"TASK 11b.2: Pressure vs Temperature for different Ball's Numbers"

# FOR DIFFERENT VALUES OF THE BALLS' NUMBER, ANLAYSES THE RELATION BETWEEN 
# KINETIC ENERGY OF THE PARTICLES (THE TEMPERATURE) AND PRESSURE

SimTN = []
nTN = [10, 20, 30]
RcTN = 50.e-3
boostTN = [1.5, 2, 3]

#1. Create multiple simulation for the different values of n and velocity 
#   (hence temperature)
for i in range(len(nTN)):
    simTN = Mod.Simulation(R_container = RcTN, N = nTN[i])
    SimTN.append(simTN)
    for j in range(len(boostTN)):
        SimTN.append(simTN.vel_increase(boostTN[j]))

T_TN = []
P_TN = []

#2.Find temperature and pressure for each
for i in range(len(SimTN)):
    SimTN[i].run(F)
    print("TN", i)
    T_TN.append(SimTN[i].temperature())
    P_TN.append(SimTN[i].pressure())
 
S = RcTN ** 2 
V = np.pi * S  

Sstring = "{:.2e}".format(S)  


#3.Plot for Ideal Gas
pl.figure(figsize = [8, 6])
pl.axes(ylim = [0, 11 / 10 * max(P_TN)])


for j in range(len(nTN)):
    Ii = j * (len(boostTN)+1)
    If = (j+1) * (len(boostTN)+1)
    Tj = T_TN[Ii : If]
    Pj = P_TN[Ii : If]
         
    data = pl.plot(Tj, Pj, "x", label = "Empirical Data (N = %g)" % (nTN[j]))
    c = data[0].get_color()
    X = np.linspace(0, 1.1 * max(Tj), 1000)
    grI = nTN[j] * Mod.Simulation.kb / V 
    pl.plot(X, grI * X, ls = "dashed", color = c, label =   "Ideal Gas Curve")
     

pl.legend(fontsize = 10)
pl.title("Pressure - Temperature Relation (V = " + Sstring + " " \
         + sympy.pretty(pi) + " m$^2$)", size = 20, family = "Times New Roman")
pl.xlabel("Temperature (K)", size = 15)
pl.ylabel("Pressure (Pa * m)", size = 15)
pl.grid()
pl.show()




"TASK 12.1: Pressure vs N for different volumes"

# FOR DIFFERENT VALUES OF THE VOLUME, ANLAYSES THE RELATION BETWEEN THE 
# PRESSURE AND THE BALLS' NUMBER, FOR A FIXED TEMPERATURE
F = 50
SimN = []
nN = np.array([10, 20, 30]) # must be from smaller to bigger
addn = nN[1:] - nN[:-1]

RcN = np.array([30, 50]) * 1e-3
SN = RcN ** 2
VN = np.pi * SN


#1. Create multiple simulation for the different values of n and Rc, 
#   but SAME TEMPERATURE
for i in range(len(nN)):
    if i == 0:
        simN = Mod.Simulation(R_container = min(RcN), N = nN[0]) 
        
    else:
        simN = simN.vel_increase(1)
        simN.add_balls(addn[i-1])
        
    for j in range(len(RcN)):
        simN = simN.vel_increase(1) # to create a copy
        simN.R_set(RcN[j])
        SimN.append(simN)
  
T_N = simN.temperature()    
P_N = []


#2.Find pressure for each
for i in range(len(SimN)):
    SimN[i].run(F)
    print("N", i)
    P_N.append(SimN[i].pressure())
   

#3.Plot
pl.figure(figsize = [8, 6])
pl.axes(ylim = [0, 11 / 10 * max(P_N)]) 

for j in range(len(RcN)): # takes the ones with same volume

    Pj = P_N[j::len(RcN)]
    Sj = SN[j]
    Vj = VN[j]
    
    Sjstring = "{:.2e}".format(Sj)

    data = pl.plot(nN, Pj, "x", label = "Empirical Data (V = " + Sjstring +\
                   " " + sympy.pretty(pi) + " m$^2$)")
    c = data[0].get_color()
    X = np.linspace(0, 1.1 * max(nN), 1000)
    grI =  Mod.Simulation.kb * T_N / Vj 
    pl.plot(X, grI * X, ls = "dashed", color = c, label =  "Ideal Gas Curve ")

      
pl.legend(fontsize = 10)
pl.title("Pressure - Balls' Number Relation (T = %g K)" % (T_N), \
         size = 20, family = "Times New Roman")
pl.xlabel("Balls' Number", size = 15)
pl.ylabel("Pressure (Pa * m)", size = 15)
pl.grid()
pl.show()




"TASK 12.2: Pressure vs Volume for different Balls' Numbers"

# FOR DIFFERENT VALUES OF THE BALLS' NUMBER, ANLAYSES THE RELATION BETWEEN 
# PRESSURE AND VOLUME, AT A FIXED TEMPERATURE

SimV = []
nV = np.array([10, 20])
addn = nV[1:] - nV[:-1]

RcV = np.array([50, 50 * np.sqrt(2), 100, 100 * np.sqrt(2)]) * 1e-3
SV = RcV ** 2
VV = np.pi * SV

#1. Create multiple simulation for the different values of n and Rc, 
#   but SAME TEMPERATURE
for i in range(len(nV)):
    if i == 0:
        simV = Mod.Simulation(R_container = min(RcV), N = nV[0])
        
    else:
        simV = simV.vel_increase(1) # to create a deepcopy
        simV.add_balls(addn[i-1])
        
    for j in range(len(RcV)):
        simV = simV.vel_increase(1) # to create a deepcopy
        simV.R_set(RcV[j])
        SimV.append(simV)
      
T_V = simV.temperature()       
P_V = []


#2.Find pressure for each
for i in range(len(SimV)):
    SimV[i].run(F)
    print("V", i)
    P_V.append(SimV[i].pressure())

  
#3.Plot
pl.figure(figsize = [8, 6])
pl.axes(ylim = [0, 1.1 * max(P_V)])

for j in range(len(nV)):
    Ii = j * (len(RcV))
    If = (j+1) * (len(RcV))
    Pj = P_V[Ii : If]
       
    data = pl.plot(SV, Pj, "x", label = "Empirical Data (N = %g)" % (nV[j]))
    c = data[0].get_color()
    XS = np.linspace(min(SV) / 10, 1.1 * max(SV), 100000)
    Num = nV[j] * Mod.Simulation.kb * T_V
    pl.plot(XS, Num / XS / np.pi, ls = "dashed", color = c, \
            label = "Ideal Gas Curve")


pl.legend(fontsize = 10)
pl.title("Pressure - Volume Relation (T = %g K)" % (T_V), \
         size = 20, family = "Times New Roman")
pl.xlabel("Volume  (" + sympy.pretty(pi) + "m$^2$)", size = 15)
pl.ylabel("Pressure (Pa * m)", size = 15)
pl.grid()
pl.show()



#%%
"Van der Waal Fitting and Plots for all the previous cases"

# IF ANY OF THE PREVIOUS CASES HADN'T BEEN EXECUTED, OR ITS VARIABLES HAD
# BEEN DELETED, THIS CELL STILL WORKS FINE, EVEN IF OF COURSE THE ANALYSIS 
# IS LESS COMPLETE

"Calculation of the b factor for the Van der Waal fit"
Sim = SimTV + SimTN + SimN + SimV

B_list = []
Rb = Sim[0].b_radius() #same for all
kb = Mod.Simulation.kb

for i in range (len(Sim)):
    
    N = Sim[i]._N
    V = Sim[i]._V
    T = Sim[i]._T
    P = Sim[i]._P
       
    Bi = ((V / N - kb * T / P ) / np.pi / Rb ** 2)
    # the smallness of Rb and the limitation of the simulation tools (after all
    # it's only a simulation right?) cause big oscillation in the value of B:
    # only the ones in the "reasonable region Bi > 0.5" are used.
    if Bi < 0.5:
        print(N, "smaller than 0")
    if Bi >= 0.5:
        B_list.append(Bi)

B = np.mean(B_list)
Berr = np.std(B_list, ddof = 1)



"Van der Waal plot against Temperature for different volumes"
if SimTV:
    pl.figure(figsize = [8, 6])
    pl.axes(ylim = [0, 1.1 * max(P_TV)])
  
    for j in range(len(RcTV)):
        Ii = j * (len(boostTV)+1)
        If = (j+1) * (len(boostTV)+1)
        Tj = T_TV[Ii : If]
        Pj = P_TV[Ii : If]
        S =  RcTV[j] ** 2
        V =  np.pi * S
            
        Sstring = "{:.2e}".format(S)
        
        data = pl.plot(Tj, Pj, "x", label = "Empirical Data (V = " + \
                       Sstring + " " + sympy.pretty(pi) + " m$^2$)")
        c = data[0].get_color()
        X = np.linspace(0, 1.1 * max(Tj), 1000)
        
        gr = nTV * Mod.Simulation.kb / V 
        pl.plot(X, gr * X, ls = "dashed", color = c, \
                label = "Ideal Gas Curve")
            
        Y = van_der_waal_Rb(nTV, V, Mod.Simulation.kb, X, B, Rb)
        pl.plot(X, Y, color = c, label = "Van der Waal Curve") 
    
    
    pl.legend(fontsize = 10)
    pl.title("Pressure - Temperature Relation (N = %g)" % (nTV), size = 20, \
             family = "Times New Roman")
    pl.xlabel("Temperature (K)", size = 15)
    pl.ylabel("Pressure (Pa * m)", size = 15)
    pl.grid()
    pl.show()
   
    

"Van der Waal Plot vs Temperature for different Balls' Numbers"
if SimTN:
    pl.figure(figsize = [8, 6])
    pl.axes(ylim = [0, 1.1 * max(P_TN)])
    S = RcTN ** 2 
    V = np.pi * S 
       
    for j in range(len(nTN)):
        Ii = j * (len(boostTN)+1)
        If = (j+1) * (len(boostTN)+1)
        Tj = T_TN[Ii : If]
        Pj = P_TN[Ii : If]
          
        data = pl.plot(Tj, Pj, "x", \
                       label = "Empirical Data (N = %g)" % (nTN[j]))
        c = data[0].get_color()
        
        X = np.linspace(0, 1.1 * max(Tj), 1000)
        grI = nTN[j] * Mod.Simulation.kb / V 
        pl.plot(X, grI * X, ls = "dashed", color = c, \
                label = "Ideal Gas Curve")
        
        Y = van_der_waal_Rb (nTN[j], V, Mod.Simulation.kb, X, B, Rb)
        pl.plot(X, Y, color = c, label =   "Van der Waal Curve") 


    pl.legend(fontsize = 10)
    pl.title("Pressure - Temperature Relation (V = " + Sstring + " " \
             + sympy.pretty(pi) + " m$^2$)", size = 20, family = "Times New Roman")
    pl.xlabel("Temperature (K)", size = 15)
    pl.ylabel("Pressure (Pa * m)", size = 15)
    pl.grid()
    pl.show()
  
   
  
"Van der Waal plot against Balls' Number for different volumes"  
if SimN:
    pl.figure(figsize = [8, 6])
    pl.axes(ylim = [0, 1.1 * max(P_N)])
    
    
    Rb = SimTN[0].b_radius()
    
    
    for j in range(len(RcN)): # takes the ones with same volume
    
        Pj = P_N[j::len(RcN)]
        Sj = SN[j]
        Vj = VN[j]
          
        Sjstring = "{:.2e}".format(Sj)
    
        data = pl.plot(nN, Pj, "x", label = "Empirical Data (V = " + \
                       Sjstring + " " + sympy.pretty(pi) + " m$^2$)")
        c = data[0].get_color()
        X = np.linspace(0, 1.1 * max(nN), 1000)
        grI =  Mod.Simulation.kb * T_N / Vj 
        pl.plot(X, grI * X, ls = "dashed", color = c, \
                label =  "Ideal Gas Curve ")
        
        Y = van_der_waal_Rb(X, Vj, Mod.Simulation.kb, T_N, B, Rb)
        pl.plot(X, Y, color = c, label = "Van der Waal Curve")
           
        
    pl.legend(fontsize = 10)
    pl.title("Pressure - Balls' Number Relation (T = %g K)" % (T_N), \
             size = 20, family = "Times New Roman")
    pl.xlabel("Balls' Number", size = 15)
    pl.ylabel("Pressure (Pa * m)", size = 15)
    pl.grid()
    pl.show()
    
    
    
"Van der Waal plot against Volume for different Balls' Numbers"
if SimV:
    pl.figure(figsize = [8, 6])
    pl.axes(ylim = (0, 1.1 * max(P_V)))

    for j in range(len(nV)):
        Ii = j * (len(RcV))
        If = (j+1) * (len(RcV))
        Pj = P_V[Ii : If]
           
        data = pl.plot(SV, Pj, "x", \
                       label = "Empirical Data (N = %g)" % (nV[j]))
        c = data[0].get_color()
        
        XS = np.linspace(min(SV) / 10, 1.1 * max(SV), 10000) 
        XV = np.pi * XS
        Id = nV[j] * kb * T_V
        pl.plot(XS, Id / XV, ls = "dashed", color = c, \
                label = "Ideal Gas Curve")
        
        XS2 = np.linspace(1.5 * nV[j] * B * Rb**2, 1.1* max(SV), 10000)
        XV2 = np.pi * XS2
        Y = van_der_waal_Rb(nV[j], XV2, kb, T_V, B, Rb)
        pl.plot(XS2, Y, color = c, label = "Van der Waal Curve")
    
    
    pl.legend(fontsize = 10)
    pl.title("Pressure - Volume Relation (T = %g K)" % (T_V), \
             size = 20, family = "Times New Roman")
    pl.xlabel("Volume  (" + sympy.pretty(pi) + "m$^2$)", size = 15)
    pl.ylabel("Pressure (Pa * m)", size = 15)
    pl.grid()
    pl.show()