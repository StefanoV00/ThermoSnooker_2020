# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 02:20:05 2020

@author: Stefano
"""

###########################################################################   
# SOME FUNCTIONS WHICH ARE USEFUL INSIDE THE SIMULATION, BUT MIGHT BE USEFUL
# OUTSIDE AS WELL

import numpy as np          
            
def k_calc (l):
    """Calculates the sum kinetic energy of a list of Ball objects"""
    k = 0
    for j in range(len(l)):
        k += l[j].kinetic() 
    return k


def mom_calc (l):
    """Calculates the sum of x-momenta, y-momenta, momenta, and 
    momenta's magnitudes of a list of Ball objects."""
    
    Spx = 0    # sum of x component of momenta
    Spy = 0    # sum of y component
    SpM = 0    # sum of magnitudes  
    for i in range(len(l)):
        m = l[i]._m
        v = l[i]._v
        vx = round (v[0], 5)
        vy = round (v[1], 5)
        Spx += (m * vx)
        Spy += (m * vy)
        SpM += (m * np.sqrt(vx ** 2 + vy ** 2))
        
    Sp = Spx + Spy #  sum of momenta
    return (Spx, Spy, Sp, SpM)


def ang_mom_calc (l): 
    """Calculates the sum of Angualr Momenta of a list of Ball objects"""
    SL  = 0     
    
    for i in range(len(l)):
        m = l[i]._m
        r = l[i]._r
        rx = round(r[0], 5 )
        ry = round(r[1], 5 )
        v = l[i]._v
        vx = round(v[0], 5 )
        vy = round(v[1], 5 )
        SL += round((m) * (rx * vy - ry * vx), 4)
    return SL
        


def vx_distribution (vx, m, kb, T): # same for y-component by symmetry
    """The one component velocity distribution as a function of the mass, 
    Boltzmann constant and temperature."""
    C = m / 2 / kb / T
    A = np.sqrt(C / np.pi)
    B = - C * vx ** 2
    return A * np.exp(B)


def max_boltz_distribution (v, m, kb, T):
    """Maxwell-Boltzmann distribution as a function of the mass, 
    Boltzmann constant and temperature."""
    C = m * v / kb / T
    B = - C * v / 2
    return C * np.exp(B)
        

def van_der_waal (N, V, kb, T, b):  
    """Calculates the pressure of a Van der Wall gas with a = 0, as a function
    of volume, Boltzmann constant, temperature, and b (volume dimensions)"""
    # returns the pressure for a Van der Waal gas with a == 0
    P = N * kb * T / (V - N * b)
    return P


def van_der_waal_Rb (N, V, kb, T, b, Rb):
    """Calculates the pressure of a Van der Wall gas with a = 0, as a function
    of volume, Boltzmann constant, temperature, the radius of the balls/
    particles and b (which is an adimensional factor)"""
    P = N * kb * T / (V - N * b * np.pi * Rb ** 2)
    return P