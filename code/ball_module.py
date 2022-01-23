# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:10:37 2020

@author: Stefano
"""

import numpy as np
import pylab as pl 

                                                                            

class Ball:
    """"A 2D ball with arguments:
        
        mass, radius, 2D position array-like, 2D velocity array-like,
        color, list of radii of other balls, if necessary."""
      
    def __init__(self, m = 1, R = 1, r=[0, 0], v=[0, 0], c = "red", rl = []):
        self._m = float(m)
        self._R = float(R)
        self._r = np.array(r, dtype='float64')
        self._v = np.array(v, dtype='float64')
        self._fc = c
        self._rlist = rl # useful to have a list of balls' R in the ball class
                         # if wanting to have multiple species of particles. 
                         # In particular, in time to collision and collide
                         # methods (see later)


        if len(r)!=2 or len(v)!=2:
            raise Exception("Wrong dimensions for r and/or v!")
            
            
    def __repr__ (self):
        return "A 2D ball (mass = %g Radius = %g, r_vector = [%g, %g], \
            v_vectorl = [%g, %g])" \
                % (self._m, self._R, self._r[0], self._r[1], \
                   self._v[0], self._v[1])
           
                
    def __str__ (self):
        return "(m = %g, R = %g, rvec = [%g, %g], vel = [%g, %g])" \
                % (self._m, self._R, self._r[0], self._r[1], \
                   self._v[0], self._v[1])
    
    
    def radius(self):
        return self._R
    
    
    def pos (self):
        return self._r


    def vel (self):
        return self._v
    
    
    def get_patch(self):
        patch = pl.Circle(self._r, self._R, ec = "blue", fc = self._fc)
        return patch
    
    
    def move (self, dt):
        self._r = self._r + self._v * dt       
        return self
    
    
    def kinetic (self):
        return self._m * np.dot(self._v, self._v) / 2
    
    
    def time_to_collision (self, other):     
        """Computes time to next collision with other ball:
        
        It takes another ball object as only argument."""
    
        dist = self._r - other._r
        dvel = self._v - other._v
        distM2 = dist[0] ** 2 + dist[1] ** 2        
        dvelM2 = dvel[0] ** 2 + dvel[1] ** 2

        A = - np.dot (dist, dvel)
        
        #Distinguishes two cases: collision to other ball or to container
        
        if other._R in self._rlist: 
            D = (A ** 2 - (distM2 - ((self._R + other._R) ** 2)) * dvelM2)
        
        else:
            D = (A ** 2 - (distM2 - ((other._R - self._R) ** 2)) * dvelM2)
        
        #It's a quadratic equation: 
            #the smallest real positive solution is taken, if there is
       
        if D < 0 or dvelM2 == 0:
            return None
        
        else:
            B = np.sqrt(D)
            dt1 = (A - B) / dvelM2
            dt2 = (A + B) / dvelM2
            
            if dt2 < 0:
                return None
            
            elif dt1 < 0:
                return dt2
            
            else:
                return min([dt1, dt2])
            
            
    def collide (self, other):
        """Performs the collision between two ball objects:
            
            takes another ball as only argument"""
        
        # Calculates (twice) kinetic energy of the two balls, 
        # to check afterwards it didn't change
        K11 = round(self.kinetic(), 5)
        K21 = round(other.kinetic(), 5) 
        K1 = K11 + K21
        
        # For the collisions bewteen equal balls (faster than general way)
        if isinstance(other, Ball) == True and other._R == self._R:
            
            dist = self._r - other._r
            distM2 = np.dot(dist, dist)
            dvel = self._v - other._v
            D = np.dot (dvel, dist) / distM2 * dist
            
            self._v = self._v - D
            other._v = other._v + D 
            # It's clear omentum is conserved
            
        # For the collisions bewteen balls in general
        elif isinstance(other, Ball) == True and other._R in self._rlist:
            
            dist = self._r - other._r
            distM2 = np.dot(dist, dist)
            dm = self._m - other._m
            sm = self._m + other._m
            
            u1p = np.dot(self._v, dist) / distM2 * dist
            u2p = np.dot(other._v, dist) / distM2 * dist
            v1p = (dm * u1p + 2 * other._m * u2p) / sm
            v2p = (2 * self._m * u1p - dm * u2p) / sm
            
            self._v = self._v - u1p + v1p
            other._v = other._v - u2p + v2p
            # Momentum is conserved
            
        # For collisions against the container's wall
        elif isinstance(other, Ball) == True: 
            
            # momentum's magnitude
            p = round(self._m * np.sqrt(np.dot(self._v, self._v)), 6) 
            
            distM2 = np.dot (self._r, self._r)
            D = np.dot (self._v, self._r) / distM2 * self._r
            
            self._v = self._v - 2 * D
            
            pnew = round(self._m * np.sqrt(np.dot(self._v, self._v)), 6)
            
            if pnew != p:
                raise Exception ("The momentum isn't conserved")
            
        # If the other object is not a ball there's a problem
        else:
            raise Exception ("Colliding with a non-ball object sir!")
        
        # If the kinetic energy of the two changed, there is a problem
        K12 = round(self.kinetic(), 5)
        K22 = round(other.kinetic(), 5)
        K2 = K12 + K22
        if K2 != K1 and type(K2) == float:
            print ("K1 = ", K1, ". ", "K2 = ", K2 )
            raise Exception ("The kinetic energy is not conserved!")  