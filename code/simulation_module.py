# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:10:37 2020

@author: Stefano
"""


import numpy as np
import copy
import pylab as pl
from scipy import stats as stats

from ball_module import * # imports Ball class
from func_module import *


"""
NOTE:
In the last days before the submission the class has been enhanced to support 
multi-specie gas analysis. T treat contemporarily both the single and multiple 
specie cases I had two options:
    - make methods with many short if loops inside
    - make methods with few extra-large if loops inside

The first I chose was the the former, but, after having seen it was sometimes
messier I used the second one for pos_hist and vel_hist methods. 

The transition to multiple-species cases was not complete, in particular the
method add_balls has not been fully updated, however, since it was an extra, I
guess it's better than nothing.'
"""


class Simulation:
    """

        Parameters
        ----------
        m : float, scalar or array-like
            The mass of the balls. The default is 1e-3.
        t : float, optional
            Time. The default is 0.
        R_container : float, optional
            Radius of container. The default is 1e-2.
        R_balls : float, scalar or array-like
            Radius of balls. The default is 1e-3.
        N : int, scalar or array-like
            Number of balls. If array-like, number of balls per specie.
            The default is 10.
        c : str, or array of strings.
            Color of balls. The default is "red".
        xboost : float, optional
            Starting elocity of balls in x-direction. The default is 0.
        yboost : float, optional
            Starting elocity of balls in x-direction. The default is 0.
        dist : float, optional
            Ratio of distance between balls and bigger balls' radius (> 0), 
            when initialising. 

        Raises
        ------
        Exception
            If lenght of m, R_balls, N don't coincide, raise expcetion. Their
            legth should be same: the number of species of balls.

    """
    
    kb = 1.381e-23 #Boltzmann Constant        
    
    def __init__(self, m = 1e-3, t = 0, R_container = 1e-2, \
                 R_balls = 1e-3, N = 10, c = "red", \
                     xboost = 0, yboost = 0, dist = 1):
          
        if hasattr(N, "__len__"):
            if len(m) != len (N) or len (N) != len(R_balls):
                raise Exception ("Wrong dimensions for m or R_balls or N")
            
        self.__t = float(t)
        self.__Rc = float(R_container)
        self.__Rb = R_balls
        
        #THE BALLS ARE CREATED AND APPENDED TO A LIST (ALSO THEIR RADII):
        
        self._blist = []
        self._Rblist = []
        self._m = m
        self.color = c
        
        # See creator method for more information 
        self.XYgrid = []   
        self.randompick = [] 
        self.creator(N, m, R_balls, R_container, xboost, yboost, c, dist) 
        self._N = len (self._blist)
        
        if type(N) == int or type(N) == list:
            self._Nj = N      # list of balls for each kind (or equal self._N)
        else: #numpy array
            self._Nj = N.tolist()

        
        self._container = Ball(100, R_container, c = "None", rl = self._Rblist) 
        # rl doesn't contain the radius of the container
        
        self._blist.append(self._container)
        self._Rblist.append(R_container)
        
        
        #SOME PHYSICAL QUANTITIES:
        
        self._A = 2 * np.pi * R_container       # 1D-AREA
        self._V = np.pi * R_container ** 2      # 2D-VOLUME
        self._P = 0                             # TOTAL PRESSURE
        self._U = k_calc (self._blist)          # INTERNAL ENERGY
        self._T = self._U / self._N / Simulation.kb   # TEMPERATURE
        
        if hasattr(self._Nj,"__len__"):
            self._Pj = [0] * len(self._Nj)       # PARTIAL PRESSURE
        
        
        #FOR NEXT COLLISION CALCULATIONS
        
        self.dt_i = [] #list of shortest time interval dt for each ith ball 
                       #before such ball collides with another
        self.jcount =[]#list of index j of the ball which collides 
                       #with ball i after dt_i[i]
        self.ib = 0    #i index of the last ball which collided
        self.jb = 0    #j index of the ball which ball i just collided against
        self._Dmom = 0 #cumulative magnitude of the variation of 
                       #momentum of particles, needed for pressure
        if hasattr(self._Nj,"__len__"):
            self._Dmomj = [0] * len(self._Nj) # partial changes in momentum
               
                            
    def __repr__ (self):
        return "Simulation at time t=%g with %g balls" %(self.__t, self._N) 
    
    def time (self):
        return self.__t
    
    def radius (self):
        return self.__Rc
    
    def b_radius (self):
        return self.__Rb
    
    def mass (self):
        return self._m
    
    def balls_number (self):
        return self._N
    
    def balls_numbers (self):
        return self._Nj
    
    def area_1D (self):
        return self._A    
    
    def volume_2D (self):
        return self._V
    
    def pressures (self):
        return self._Pj
    
    def pressure (self):
        return self._P
    
    def int_energy (self):
        return self._U
    
    def temperature (self):
        return self._T
       
   
    
    def next_collision (self):
      
        """Moves the simulation to the next collision, advances in time, 
        calculates the variation in pressure on the container, 
        performs such collision. Only argument is self."""
        
        dt_ij = [] # needed later
        
        
        #1. CALCULATE THE SMALLEST INTERVAL OF TIME (dt) BEFORE NEXT COLLISION
        
        # If it is the very first time, a longer time is needed
        if  type(self.dt_i) == list: # after 1st time, dt_i becomes np.array
            blist2 = copy.deepcopy(self._blist)  
            
            for i in range(self._N): # skips the container
                del blist2[0]
                j = 0
                shift = 0
                while j < len(blist2):
                    # caculates the dtij: the time interval before collision 
                    # between ith ball and jth ball. 
                    # Does so for a specific i for all j,
                    # and appends values to a list dt_ij
                    dtij =  self._blist[i].time_to_collision(blist2[j])
                    if not dtij:
                        shift += 1
                    else:
                        dt_ij.append(dtij) 
                    j += 1 
                
                if dt_ij:

                    self.dt_i.append(min(dt_ij))  
                    self.jcount.append(dt_ij.index(min(dt_ij)) + i + shift + 1) 
                    dt_ij = [] 
                
                elif not dt_ij: # if the ball is stuck 
                   # appends the maximum dt_i +10, so that for sure it won't be
                   # the minimum
                    self.dt_i.append(max(self.dt_i)+10)  
                    # append itself (it doesn't really matter)
                    self.jcount.append(i) 
                    dt_ij = [] 
                    
                
            self.dt_i = np.array(self.dt_i) #dt_i is transformerd into array 
            
        elif self.jb == self._N: #last collision was ball vs container
            i = self.ib
            blist2 = copy.deepcopy(self._blist)
            del blist2[i]
            j = 0
            D = max(self.dt_i) + 10
            Dth_j = self.ib
            
            for j in range(self._N): 
                #iterates over all bodies (container included) apart ib^th
                a = self._blist[i].time_to_collision(blist2[j])
                if not a:
                    pass

                elif a < D:
                    D = a 
                    if j < i:
                        Dth_j = j 
                    else:         # j >= i
                        Dth_j = j + 1                                       
           
            self.dt_i[i] = D  
            self.jcount[i] = Dth_j
            D = 0
        
        else: 
        
           for i in [self.ib, self.jb]: #last collision was ball vs ball
           
                blist2 = copy.deepcopy(self._blist)
                del blist2[i]
                j = 0
                D = max(self.dt_i) + 10
                Dth_j = i
                
                for j in range(self._N): 
                    #iterates all bodies (container included) apart i^th
                    a = self._blist[i].time_to_collision(blist2[j])
                    
                    if not a:
                        pass

                    elif a < D:
                        D = a 
                        if j < i:
                            Dth_j = j 
                        else:         # j >= i
                            Dth_j = j + 1 

                self.dt_i[i] = D  
                self.jcount[i] = Dth_j
                D = max(self.dt_i) + 10
               
        # Save the shortest dt among all, its i and its j indexes, 
        # such that dt is the time interval before the ith and jth ball collide

        dt = min(self.dt_i) - (1e-6)   
        # the term must be subtracted to avoid balls screwing up
        self.ib = np.argmin(self.dt_i) 
        # saves the indexes
        self.jb = self.jcount[self.ib] 
        # Note: ib is necessarily smaller than jb
        
        
        self.__t += dt 
        self.dt_i -= dt #all the dtis are reduced of the interval dt
        
        
        
        #2.MOVE THE PARTICLES OF THE AMOUNT OF TIME dt
        
        for i in range(self._N): #skips the container, as it doesn't move
            self._blist[i].move(dt)
        
        
        
        #3. CALCULATE VARIATION IN BOTH TOTAL AND PARTIAL PRESSURE
        # obviously only executes if one ball is impacting the container
        
        #3.1 TOTAL PRESSURE 
        if self.jb == self._N:
            b = self._blist[self.ib]
            Dvel = 2 *  np.dot (b._v, b._r) / np.dot (b._r, b._r) * b._r
            Dmom_vec = abs (b._m * Dvel)
            self._Dmom += np.sqrt(np.dot (Dmom_vec, Dmom_vec))
            self._P = self._Dmom / self.__t / self._A 
           
            #3.2 PARTIAL (actually this pressures are computed after run method
            # here it only compute the Dmomj, see line 462)
            if hasattr(self._Nj,"__len__"):
                M = np.array(self._m)
                R = np.array(self.__Rb)
                j1 = np.where(M == b._m)[0]
                j2 = np.where(R == b._R)[0]
                for k in j1:
                    if k in j2:
                        j = k
                        break
                self._Dmomj[j] += np.sqrt(np.dot (Dmom_vec, Dmom_vec))
                
                
        #4.MAKE THE COLLISION HAPPEN  
        self._blist[self.ib].collide(self._blist[self.jb])
    
    
    
    def creator (self, Nj, m, Rbj, Rc, xboost, yboost, color, dist): 
        """Creates a grid were to spawn the balls, and creates the balls
        in random positions of the grid."""

        # For the different cases (single or multiple species of particles):
            # uniforms the definitions of Nj, m, Rb, color.

        if hasattr(Nj, "__len__"):
            N = sum(Nj)
            Rb = max(Rbj)
            
            M = []
            RB = []
            Col = []
            for j in range(len(Nj)):
                mj = [m[j]] * Nj[j]
                Rbjj = [Rbj[j]] * Nj[j]
                if hasattr(color, "__len__"):
                    col = [color[j]] * Nj[j]
                else:
                    col = [color] * Nj[j]
                M += mj
                RB += Rbjj
                Col += col
        else:
            N = Nj
            Rb = Rbj
            RB = [Rbj] * Nj
            M = [m] * Nj
            Col = [color] * Nj
            
        # 2. CREATES A GRID WHERE TO PLACE THE BALLS WHICH MUST BE CREATED
        
        X2 = np.arange(start = 0, stop = Rc - Rb, step = (2 + dist) * Rb )
        Y2 = np.arange(0, Rc - Rb, (2 + dist) * Rb )
        X1 = - np.delete(X2, 0)
        Y1 = - np.delete(Y2, 0)
        X = np.concatenate ((X1, X2))
        Y = np.concatenate ((Y1, Y2))
        
        self.XYgrid = []
        for i in range(len(X)):
            for j in range(len(Y)):
                if np.sqrt(X[i]**2 + Y[j]**2) + Rb <= Rc:
                    self.XYgrid.append([X[i],Y[j]])
                    
        if len(self.XYgrid) < N:
            
            # len(XYgrid) rpresents the max number (Nmax) of balls 
            # which fit the simulation
            # if N bigger than Nmax, N is fixed equal to Nmax:
            # the simulation keeps running and a warning is printed
            # to inform the user
            
            print ("I'm sorry, but you wanted too many balls to be created: \
                   N = %g is the very maximum for this container"  \
                       %  (len(self.XYgrid)))
            N = len(self.XYgrid) 
            
            
        # 3. NOW RANDOMLY PICKS POSITIONS FROM GRID WHERE TO "SPAWN" BALLS    
        
        self.randompick = np.random.choice(len(self.XYgrid),N,replace = False)
        
        
        # 4. CREATES THE BALLS
        
        for i in range(N):
            r = self.XYgrid[self.randompick[i]]
            b = Ball(M[i], RB[i], r, \
                     [np.random.normal(xboost), np.random.normal(yboost)], \
                         c = Col[i], rl = RB [:N])
            self._blist.append(b)
            self._Rblist.append(Rb)
            
            
         
            
    def run(self, num_frames, animate = False, period = 0.5, \
            plot_p = False, plot_K = False, plot_L = False):
        
        """Runs the simulation and, if asked to, plots the animation,
        plots graphs of Momentum and kinetic Energy and Angular Momentum."""
        
        #RUNS THE SIMULATION AND:
            #IF ANIMATE IS TRUE, DISPLAYS THE ANIMATION
            #IF PLOT_P IS TRUE, PLOTS MOMENTUM'S CHARACTERISTICS OVER TIME
        
        if animate:
            
            pl.figure(figsize = [7, 7])
            L = 1.1 * self.__Rc 
            ax = pl.axes(xlim=(-L, L), ylim=(-L, L))
            
            #Sets the title and some nice text 
            ax.set_title("ThermoSnooker Simulation", c = "r", \
                         size = 20, family = "Times New Roman")
            ax.ticklabel_format(axis = "both", style = 'scientific')
            ax.text( - self.__Rc, 1.04 * self.__Rc, "N = %g" % (self._N))
            
            if not hasattr(self._Nj, "__len__"): 
                ax.text( L, 1.04 *self.__Rc,"Rb = %g m" % (self.__Rb), \
                        ha = "right")
                ax.text( L, 0.97 *self.__Rc,"m = %g kg" % (self._m), \
                        ha = "right")
            else:
                ax.text(L, 1.04 *self.__Rc, "Rb =(" \
                        + str(self.__Rb).strip("[]") + ")m", ha = "right")
                ax.text(L, 0.97 *self.__Rc, "m =(" \
                        + str(self._m).strip("[]") + ")kg", ha = "right")
                    
            ax.add_artist(self._container.get_patch())
            text1 = ax.text( - self.__Rc, 0.97 * self.__Rc, "")
            text2 = ax.text( - self.__Rc, 0.90 * self.__Rc, "")
            ax.set_xlabel("metres", size = 15)
            ax.set_ylabel("metres", size = 15)
            for i in range(self._N):
                ax.add_patch(self._blist[i].get_patch())
        
        if plot_p:           
            Spx = []    # list to append the sum of x component of momenta 
            Spy = []    # list to append the sum of y component
            Sp = []     # list to append the sum of momenta
            SpM = []    # list to append the sum of magnitudes 
            tp   = []    # list of time values
        
        if plot_L:
            SL  = []    #list to append the sum of ang. momenta
            tL   = []    # list of time values
            

        for frame in range(num_frames):
            self.next_collision() # runs the simulation
            
            if animate:
                ax.patches = []
                text1.set_text("Frame: %g "  % (frame))
                text2.set_text("Time: %g s" % (self.__t))      
                for i in range(self._N):
                    ax.add_patch(self._blist[i].get_patch())
                pl.pause(period) 
            
            if plot_p:
                x, y, S, M = mom_calc(self._blist) 
                Spx.append(x)    
                Spy.append(y)    
                Sp.append(S)
                SpM.append(M)
                tp.append(self.__t) 
            
            if plot_L:
                l = ang_mom_calc(self._blist)
                SL.append(l)
                tL.append(self.__t)
                
                           
            if frame == num_frames // 2: # just to know it's working fine
                print("We are at half the running time of this simulation")
        
        #Computes the partial pressures
        if hasattr(self._Nj, "__len__"):
            self._Pj = np.array(self._Dmomj) / self.__t / self._A 
        
        if animate:
            
            pl.show()
        
        if plot_p:
            fig, axs = pl.subplots(2, 2, sharex=True, sharey=True, \
                                   figsize = (8,8))   

            #1
            axs[0,0].plot(tp, Spx)
            axs[0,0].set_ylabel("Momentum (m/s)", size = 13)
            axs[0,0].set_title("Sum of Momenta's x-components", size = 17, \
                               family = "Times New Roman")
            axs[0,0].grid()
            axs[0,0].annotate("a)", (-0.2, 1.05), xycoords = "axes fraction")
            #2
            axs[0,1].plot(tp, Spy)
            axs[0,1].set_title("Sum of Momenta's y-components", size = 17, \
                               family = "Times New Roman")
            axs[0,1].grid()
            axs[0,1].annotate("b)", (-0.13, 1.05), xycoords = "axes fraction")
            #3
            axs[1,0].plot(tp, Sp)
            axs[1,0].set_xlabel("Time (s)", size = 13)
            axs[1,0].set_ylabel("Momentum (m/s)", size = 13)
            axs[1,0].set_title("Sum of Momenta", size = 17, \
                               family = "Times New Roman")
            axs[1,0].grid()
            axs[1,0].annotate("c)", (-0.2, 1.05), xycoords = "axes fraction")
            #4
            axs[1,1].plot(tp, SpM)
            axs[1,1].set_xlabel("Time (s)", size = 13)
            axs[1,1].set_title("Sum of Momenta's Magnitudes", size = 17, \
                               family = "Times New Roman")
            axs[1,1].grid()
            axs[1,1].annotate("d)", (-0.13, 1.05), xycoords = "axes fraction")
            
            fig.tight_layout()
            fig.show()
        
        if plot_K:
            # a plot of kinetic energy in time: a constant line, 
            # as if it were not constant, the simulation would raise exception 
            pl.figure()
            pl.plot(np.linspace(0, self.__t, 1000), [self._U] * 1000 )
            pl.title("The (Constant) Internal Energy of the System", \
                     size = 17, family = "Times New Roman")  
            pl.xlabel("Time (s)", size = 12)
            pl.ylabel("Sum of Kinetic Energies (J)", size = 12)
            pl.tight_layout()
            pl.grid()
            pl.show() 
        
        if plot_L:
            pl.figure()  
            pl.autoscale(enable = False)
            L1 = min(SL) - abs(min(SL)) * 0.5
            L2 = max(SL) + abs(max(SL)) * 0.5
            pl.axes(xlim = (0, 1.1 * max(tL)), ylim = (L1, L2))
            pl.plot(tL, SL)
            pl.title("Sum of Angular Momenta", size = 17, \
                               family = "Times New Roman")
            pl.ylabel("Angular Momentum (kg m$^2$/s)", size = 12)
            pl.xlabel("Time (s)", size = 12)
            pl.tight_layout()
            pl.grid()
            
            pl.show()
            
                        
            
                        
    def plot(self): 
        
        """Plots the simulation (balls and container) at its current state"""
        
        pl.figure(figsize = [6.4, 6.4])
        L = 1.1 * self.__Rc
        
        #Sets axes, title and some text
        ax = pl.axes(xlim=(-L, L), ylim=(-L, L))
        ax.set_title("ThermoSnooker Simulation", c = "r", \
                     size = 20, family = "Times New Roman")
        ax.text( - self.__Rc, 1.04 * self.__Rc, "N = %g" % (self._N))
        
        if not hasattr(self._Nj, "__len__"): 
                ax.text(- self.__Rc, 0.97 * self.__Rc,"Rb = %g m" %(self.__Rb))
                ax.text( - self.__Rc, 0.90 * self.__Rc,"m = %g kg" % (self._m))
        else:
            ax.text(L, 1.04 *self.__Rc, "Rb =(" \
                    + str(self.__Rb).strip("[]") + ")m", ha = "right")
            ax.text(L, 0.97 *self.__Rc, "m =(" \
                    + str(self.__Rb).strip("[]") + ")kg", ha = "right")
                    
        ax.set_xlabel("metres", size = 15)
        ax.set_ylabel("metres", size = 15)
                    
        ax.add_artist(self._container.get_patch())
        for i in range(self._N):
            ax.add_patch(self._blist[i].get_patch())

        pl.show()            
         
          
         
    def pos_hist (self, multi = False):
        
        """ Plots the histogram of the balls' distances from the centre
        and theoretical curve. 
        
        If there are multiple species of particles and multi is True, it also
        plots the distributions of the multiple species superoposed to each
        other."""
        
        # There is no point in drawing the distributions separately if they 
        # have the same color
        if not hasattr(self.color, "__len__"):
            multi = False
        
        s = self.__Rc / 10
        R = self.__Rc + s
        rho = self._N / self._V
        bins = np.arange(0, R, s)
        r0 = []
        
        pl.figure()
        
        #1st CASE: ONLY TOTAL DISTRIBUTION
        if not hasattr(self._Nj, "__len__") or not multi:
        
            
            for i in range (self._N): #ignores the container
                r_i = self._blist[i].pos()
                r0.append (np.sqrt(np.dot(r_i, r_i)))
                     
            n, bins, patches1 = pl.hist(r0, bins = bins)
            bins_centre = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
            
                
        #2nd CASE: TOTAL DISTRIBUTION + SEPARATED DISTRIBUTIONS
        else:
            
            # Still the total distribution
            for i in range (self._N): 
                r_i = self._blist[i].pos()
                r0.append (np.sqrt(np.dot(r_i, r_i)))
                     
            n, bins, patches1 = pl.hist(r0, bins = bins)
            bins_centre = (np.array(bins[:-1]) + np.array(bins[1:])) / 2

            # The multiple distributions are superposed to the total one    
            r0 = [[] for x in range(len(self._Nj))]
            for i in range(self._N):
                for j in range(len(self._Nj)):
                    if self._blist[i]._m == self._m[j] \
                        and self._blist[i]._R == self.__Rb[j]:
                        r_ij = self._blist[i].pos()
                        r0[j].append (np.sqrt(np.dot(r_ij, r_ij)))

            nj, bins, patches2 = pl.hist(r0, bins = bins, color = self.color,\
                                        histtype = "step")
                
            # a useless 1 dot point to have a label as I like it
            # (preferred to pl.text)
            for j in range(len(self._Nj)):
                pl.plot(0, 0, label = "m = %g kg, Rb = %g m, N = %g" \
                    % (self._m[j], self.__Rb[j], self._Nj[j]))
       
        # The following is the same for both        
        if sum(n) != self._N:
            raise Exception ("%g balls are missing " % (self._N - sum(n)))
            
        #Performs the chisquare test
        expected = 2 * np.pi * bins_centre * s * rho
        Chi2, pvalue = stats.chisquare(n, expected)
            
        #Performs chisquare test and prepares the plotting
        expected = 2 * np.pi * bins_centre * s * rho
        Chi2, pvalue = stats.chisquare(n, expected)
        r = np.linspace(0, self.__Rc, 1000) 
        y = 2 * np.pi * r * s * rho 
        
        #Plots  
        pl.plot(r, y, label = "Theoretical Linear Increase \n \
p-value = %g" % (round(pvalue, 5)))
        pl.legend(fontsize = 12, loc = "upper left")
        pl.xticks(bins)
        pl.xlabel("Distance from the centre (m)", size = 12)
        pl.ylabel("Balls' Number", size = 12)
        pl.title("Balls' Distance from Centre", size = 17, \
                 family = "Times New Roman")
        
        pl.tight_layout()     
        pl.show()
        
        return (Chi2, pvalue)
        
        
        
    def inter_pos_hist (self):
        
        """ Plots the histogram of the inter-balls separation and fit"""
        
        r0 = []
        blist2 = copy.deepcopy(self._blist)
        
        for i in range (self._N): #computes separations, ignores the container 
            r_i = self._blist[i].pos()      
            del blist2[0]
            for j in range(len(blist2) - 1):
                r_ij = self._blist[j].pos()- r_i
                r0.append (np.sqrt(np.dot(r_ij, r_ij)))
        
        s = self.__Rc / 10
        R = self.__Rc + s
        bins = np.arange(0, 2 * R, s)
        
        pl.figure()
        n, bins, patches = pl.hist(r0, bins = bins)
        
        N = self._N * (self._N - 1) / 2
        if sum(n) != N:
            raise Exception ("%g balls are missing " % (N - sum(n)))

        pl.xlabel("Relative Distance (m)", size = 12)
        pl.ylabel("Balls' Number", size = 12)
        pl.title("Inter-Balls' Separation", size = 17, \
                 family = "Times New Roman")
        pl.text(0,  max(n), "width = %g m" % (s))
        
        pl.tight_layout()
        pl.show()
    
    
    
    def v_distribution(self, vx = True, vy = True, vMB = True, multi = False):
        
        """Plots the distribution of the x and y velocity and the speed,
        comparing them to the theoretical predictions.
        
        If there are multiple species of particles and multi is True, it also
        plots the distributions of the multiple species superoposed to each
        other."""
        
        # If vx is true, plots the x-component of velocity 
        # If vy is true, plots the y-component of velocity 
        # If vMB is true, plots the speed
        
        # There is no point in drawing the distributions separately if they 
        # have the same color or simply there's one specie
        if not hasattr(self.color, "__len__") \
            or not hasattr(self._Nj, "__len__"):
            multi = False
        
        # First creates the velocities and speed arrays, and if multi,
        # besides the general one, each specie has its own
        V = []
        
        if hasattr(self._Nj, "__len__") and multi:
            velxj = [[] for x in range(len(self._Nj))]
            velyj = [[] for x in range(len(self._Nj))]
            speedj = [[] for x in range(len(self._Nj))]
        
        for i in range(self._N):
            v = self._blist[i]._v
            V.append(v) 
            if hasattr(self._Nj, "__len__") and multi:
                for j in range(len(self._Nj)):
                    if self._blist[i]._m == self._m[j] \
                        and self._blist[i]._R == self.__Rb[j]:
                        velxj[j].append (v[0])
                        velyj[j].append (v[1])
                        speedj[j].append(np.sqrt(np.dot(v, v)))         
        V = np.array(V)
                   
        # Now creates subplots
        if vx and vy and vMB:
            f = 3
        elif (vx and vy) or (vx and vMB) or (vy and vMB):
            f = 2
        else:
            f = 1
        c = 0
        
        fig, axs = pl.subplots(f, 1, sharex=True, sharey=True, \
                               figsize = (8,8))

        if vx:
            
            c += 1
            velx = V[:,0]
            
            # Apply Freedman - Diaconis rule for optimal number of bins
            IQR = stats.iqr(velx)
            w = 2 * IQR / (self._N)**(1/3)
            Nb = int((max(velx) - min(velx)) // w)
            
            # Calculate the total Histogram
            pl.subplot(f, 1 ,c)
            n1, bins, patches = pl.hist(velx, bins = Nb)
            bins_centre = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
            w = bins_centre[1] - bins_centre[0]
            
            # Prepare plotting
            v = np.linspace(min(velx), max(velx), 1000)
            
            # Prepare for Chi2 test (only for bins with n > 5)
            counter = []
            for i in range(len(n1)):
                if n1[i] < 5:
                    counter.append(i)
            n1chi = np.delete(n1, counter)
            b1chi = np.delete(bins_centre, counter)
            
            # Separate the different cases
           
            if not hasattr(self._Nj, "__len__"):
                # Even if multi was off, there still would be different masses
                # therefore, this time, it goes separately
                
                # Plots
                p = self._N * w * vx_distribution(v, self._m, \
                                                     Simulation.kb, self._T)
                
                # Performs the chisquare test        
                expected = self._N * w * vx_distribution(b1chi,\
                                            self._m, Simulation.kb, self._T )
                Chi21, pvalue1 = stats.chisquare(n1chi, expected)
                
                pl.plot(v, p, label = "Gaussian Distribution\n \
p-value = %g" % (round(pvalue1, 5)))
                    
            elif not multi:
                # We still want a unifrom histogram, but, as there are multiple
                # masses, the distributions have to be computed separately
                
                p = 0
                expected = 0
                
                for j in range(len(self._Nj)):
                    # for plot
                    p += self._Nj[j] * w * vx_distribution (v, \
                                          self._m[j], Simulation.kb, self._T)

                    # for chisquare
                    expected += self._Nj[j] * w * vx_distribution(b1chi,\
                                          self._m[j], Simulation.kb, self._T)
                  
                # Performs the chisquare test and plots       
                Chi21, pvalue1 = stats.chisquare(n1chi, expected)
                pl.plot(v, p, label = "Gaussian Distribution\n \
p-value = %g" % (round(pvalue1, 5)))
        
            else: # if both hasattr and multi
                # Cancel the total histogram and plot the new ones
                patches.remove()
                pl.hist(velxj, bins = Nb, color = self.color, stacked = True)
                
                p = [0 for x in range(len(self._Nj))]
                pj = 0
                expected = 0
                
                for j in range(len(self._Nj)):
                    # for plot
                    pj += self._Nj[j] * w * vx_distribution (v, \
                                          self._m[j], Simulation.kb, self._T)
                    # for chisquare
                    expected += self._Nj[j] * w * vx_distribution(b1chi,\
                                          self._m[j], Simulation.kb, self._T)
                        
                    # The jth term is the sum of the previous's plus its own    
                    p[j] = pj

                    pl.plot(v, p[j], color = "black")
                    pl.plot(0, 0, color = self.color[j],\
                        label = "m = %g kg, Rb = %g m, N = %g" \
                                %(self._m[j], self.__Rb[j], self._Nj[j]))
                pl.plot(0, 0, color = "black",\
                    label = "Corresponding Stacked Gaussians")
                        
                Chi21, pvalue1 = stats.chisquare(n1chi, expected)

            pl.legend(fontsize = 12, loc = "upper right")
            pl.title("x-Velocity Distribution", size = 17,\
                     family = "Times New Roman")
            pl.ylabel("Number of balls", size = 12)
            pl.xlabel("Velocity's x-component (m/s)", size = 12)
            pl.annotate("a)", (-0.05, 1.05), xycoords = "axes fraction")
            
            fig.tight_layout()
            pl.show()
            
           
            
        if vy:
            c += 1
            vely = V[:,1]
            
            # Apply Freedman - Diaconis rule for optimal number of bins
            IQR = stats.iqr(vely)
            w = 2 * IQR / (self._N)**(1/3)
            Nb = int((max(vely) - min(vely)) // w)
            
            # Calculate the total Histogram
            pl.subplot(f, 1 ,c)
            n2, bins, patches = pl.hist(vely, bins = Nb)
            bins_centre = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
            w = bins_centre[1] - bins_centre[0]
            
            # Prepare plotting
            v = np.linspace(min(vely), max(vely), 1000)
            
            # Prepare for Chi2 test (only for bins with n > 5)
            counter = []
            for i in range(len(n2)):
                if n2[i] < 5:
                    counter.append(i)
            n2chi = np.delete(n2, counter)
            b2chi = np.delete(bins_centre, counter)
            
            # Separate the different cases
           
            if not hasattr(self._Nj, "__len__"):
                # Even if multi was off, there still would be different masses
                # therefore, this time, it goes separately
                
                # Plots
                p = self._N * w * vx_distribution(v, self._m, \
                                                     Simulation.kb, self._T)
                
                # Performs the chisquare test        
                expected = self._N * w * vx_distribution(b2chi,\
                                            self._m, Simulation.kb, self._T )
                Chi22, pvalue2 = stats.chisquare(n2chi, expected)
                
                pl.plot(v, p, label = "Gaussian Distribution\n \
p-value = %g" % (round(pvalue2, 5)))
                    
            elif not multi:
                # We still want a unifrom histogram, but, as there are multiple
                # masses, the distributions have to be computed separately

                p = 0
                expected = 0
                
                for j in range(len(self._Nj)):
                    # for plot
                    p += self._Nj[j] * w * vx_distribution (v, \
                                          self._m[j], Simulation.kb, self._T)

                    # for chisquare
                    expected += self._Nj[j] * w * vx_distribution(b2chi,\
                                          self._m[j], Simulation.kb, self._T)
                  
                # Performs the chisquare test and plots       
                Chi22, pvalue2 = stats.chisquare(n2chi, expected)
                pl.plot(v, p, label = "Gaussian Distribution\n \
p-value = %g" % (round(pvalue2, 5)))
        
            else: # if both hasattr and multi
                # Cancel the total histogram and plot the new ones
                patches.remove()
                pl.hist(velyj, bins = Nb, color = self.color, stacked = True)
                
                p = [0 for x in range(len(self._Nj))]
                pj = 0
                expected = 0
                
                for j in range(len(self._Nj)):
                    # for plot
                    pj += self._Nj[j] * w * vx_distribution (v, \
                                          self._m[j], Simulation.kb, self._T)
                    # for chisquare
                    expected += self._Nj[j] * w * vx_distribution(b2chi,\
                                          self._m[j], Simulation.kb, self._T)
                        
                    # The jth term is the sum of the previous's plus its own    
                    p[j] = pj
                    
                    pl.plot(v, p[j], color = "black")
                    pl.plot(0, 0, color = self.color[j],\
                        label = "m = %g kg, Rb = %g m, N = %g" \
                                %(self._m[j], self.__Rb[j], self._Nj[j]))
                pl.plot(0, 0, color = "black",\
                    label = "Corresponding Stacked Gaussians")
                        
                Chi22, pvalue2 = stats.chisquare(n2chi, expected)
                    

            pl.legend(fontsize = 12, loc = "upper right")
            pl.title("y-Velocity Distribution", size = 17,\
                     family = "Times New Roman")
            pl.ylabel("Number of balls", size = 12)
            pl.xlabel("Velocity's y-component (m/s)", size = 12)
            pl.annotate("a)", (-0.05, 1.05), xycoords = "axes fraction")
            
            fig.tight_layout()
            pl.show()

         
            
        if vMB:
            
            c += 1
            speed = []
            for i in range (self._N):
                speed.append(np.sqrt(np.dot(V[i], V[i])))
               
            # Apply Freedman - Diaconis rule for optimal number of bins
            IQR = stats.iqr(speed)
            w = 2 * IQR / (self._N)**(1/3) 
            Nb = int((max(speed) - min(speed)) // w)
            
            # Calculate Histogram and curve
            pl.subplot(f, 1 ,c)
            n3, bins, patches = pl.hist(speed, bins = Nb)
            bins_centre = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
            w = bins_centre[1] - bins_centre[0]
            
            # Prepare plotting
            v = np.linspace(min(speed), max(speed), 1000)
            
            # Prepare for Chi2 test (only for bins with n > 5)
            counter = []
            for i in range(len(n3)):
                if n3[i] < 5:
                    counter.append(i)
            n3chi = np.delete(n3, counter)
            b3chi = np.delete(bins_centre, counter)
             
            #The three cases as before
            
            if not hasattr(self._Nj, "__len__"):
                p = self._N * w * max_boltz_distribution(v, self._m, \
                                                     Simulation.kb, self._T)
                expected = self._N * w * max_boltz_distribution(b3chi,\
                                            self._m, Simulation.kb, self._T )
                    
                Chi23, pvalue3 = stats.chisquare(n3chi, expected)
                pl.plot(v, p, label = "2D Maxwell-Boltzmann Disribution\n \
p-value = %g" % (round(pvalue3, 5)))
                
            elif not multi:

                p = 0
                expected = 0
                for j in range(len(self._Nj)):
                    p += self._Nj[j] * w * max_boltz_distribution(v, \
                                          self._m[j], Simulation.kb, self._T)
                    expected +=self._Nj[j] * w * max_boltz_distribution(b3chi,\
                                            self._m[j], Simulation.kb, self._T)
                    
                Chi23, pvalue3 = stats.chisquare(n3chi, expected)
                pl.plot(v, p, label = "2D Maxwell-Boltzmann Disribution\n \
p-value = %g" % (round(pvalue3, 5)))
                    
            else:
                patches.remove()
                pl.hist(speedj, bins = Nb, color = self.color, stacked = True)
                
                p = [0 for x in range(len(self._Nj))]
                pj = 0
                expected = 0
                for j in range(len(self._Nj)):
                    pj += self._Nj[j] * w * max_boltz_distribution(v, \
                                          self._m[j], Simulation.kb, self._T)
                    expected +=self._Nj[j] * w * max_boltz_distribution(b3chi,\
                                          self._m[j], Simulation.kb, self._T)
                    p[j] = pj
            
                    pl.plot(v, p[j], color = "black")
                    pl.plot(0, 0, color = self.color[j],\
                    label = "m = %g kg, Rb = %g m, N = %g" \
                        %(self._m[j], self.__Rb[j], self._Nj[j]))
                pl.plot(0, 0, color = "black",\
                    label = "Corresponding Stacked Maxwellians")
                
                Chi23, pvalue3 = stats.chisquare(n3chi, expected)
            
      
            pl.legend(fontsize = 12, loc = "upper right")
            pl.title("Speed Distribution", size = 17, \
                     family = "Times New Roman")
            pl.ylabel("Number of balls", size = 12)
            pl.xlabel("Speed (m/s)", size = 12)
            pl.annotate("c)", (-0.05, 1.05), xycoords = "axes fraction")
            
            fig.tight_layout()
            pl.show()
        
            if sum(n3) != self._N:
                raise Exception ("Some balls are missing ")
            
        # Now returns
        if vx and vy and vMB:
            return [Chi21, pvalue1], [Chi22, pvalue2], [Chi23, pvalue3]
        elif vx and vy:
            return [Chi21, pvalue1], [Chi22, pvalue2]
        elif vx and vMB:
            return [Chi21, pvalue1], [Chi23, pvalue3]
        elif vy and vMB:
            return [Chi22, pvalue2], [Chi23, pvalue3]
        elif vx:
            return [Chi21, pvalue1]
        elif vy:
            return [Chi22, pvalue2]
        elif vMB:
            return [Chi23, pvalue3]
               
        
    
    def vel_increase (self, n = 2):
        
        """Creates a copy of the simulation with velocity n times bigger"""
        
        sim2 = copy.deepcopy(self)
        
        for j in range(self._N):
            sim2._blist[j]._v *= n
                
        sim2._U = k_calc (sim2._blist)
        sim2._T = sim2._U / sim2._N / Simulation.kb
        
        return sim2
    
    
    
    def R_increase (self, n = 2):
        
        """Makes the raidus of the chosen simulation n times bigger"""
        
        self.__Rc = self.__Rc * n
        self._container = Ball(100, self.__Rc, c = "None", \
                               rl = self._Rblist[:-1]) 
        self._A = n * self._A       
        self._V = n ** 2 * self._V
        
        self._blist[-1] = self._container
        self._Rblist[-1] = self._container
        
        for i in range(self._N):
            r = self._blist[i]._r
            dist = np.sqrt(np.dot(r, r))
            if dist + self._blist[i]._R > self.__Rc:
                raise Exception("Too small!")
        
        
        
    def R_set (self, R = 1):
        
        """Sets the raidus of the chosen simulation to R"""
        
        self.__Rc = R
        self._container = Ball(100, self.__Rc, c = "None",  \
                               rl = self._Rblist[:-1]) 
        self._A = 2 * np.pi * R       
        self._V = np.pi * R ** 2
        
        self._blist[-1] = self._container
        self._Rblist[-1] = self._container
        
        for i in range(self._N):
            r = self._blist[i]._r
            dist = np.sqrt(np.dot(r, r))
            if hasattr(self.__Rb, "__len__"):
                if dist + max(self.__Rb) > self.__Rc:
                    raise Exception("Too small!")
            else:
                 if dist + self.__Rb > self.__Rc:
                    raise Exception("Too small!")
                
                
                
    
    def add_balls (self, N = 1, m = 1e-3, Rb = 1e-3, color = "red"):
        
        """Adds n balls to the simulation, without altering the temperature. 
        
        It is RECOMMENDED TO ONLY ADD ONE TYPE OF BALL: the one which is
        already present in the gas.
        
        DON'T ADD BALLS AFTER THE SIMULATION HAS BEEN RUN: balls may superpose
        
        DON'T ADD SMALLER BALLS (smaller radii): the grid supports the spawn
        of balls with smaller radii, bigger ones would cause some balls to
        superpose.
        
        Note: it doesn't update the attributes self._Nj, self._m, self.__Rb,
        therefore if a different kind of balls has been added
        DON'T RUN AFTERWARDS.'
        
        The reader may wonder about the utility of the method at this point:
        to add balls of the same kind without altering the temperature when
        investigating the ideal gas and van der wall equations of state.
        """
        
        #1. uniformise cases in which n and others are numbers and lists
        if hasattr(N, "__len__"):
            n = sum(N)
            
            M = []
            RB = []
            Col = []
            for j in range(len(N)):
                mj = [m[j]] * N[j]
                Rbjj = [Rb[j]] * N[j]
                if hasattr(color, "__len__"):
                    colorj = [color[j]] * N[j]
                else:
                    colorj = [color] * N[j]
                M += mj
                RB += Rbjj
                Col += colorj
        else:
            n = N
            RB = [Rb] * n
            M = [m] * n
            Col = [color] * n
        
        RBtot = self._Rblist[:-1] + RB
        
        # Defines the available spots (grid) and spawns the new balls
        available = len(self.XYgrid) - self._N
        
        if  available < n:
            raise Exception ("I'm sorry, but the maximum number of balls \
                             which can be added is %g" %(available))
        
        grid = np.array(self.XYgrid)
        grid = np.delete(self.XYgrid, self.randompick, axis = 0)
        
        randompick2 = np.random.choice(len(grid), n, replace = False)
        del self._blist[-1]
        del self._Rblist[-1]
        
        KEav = self._U / self._N
        v2av_component = KEav / np.array(M)
        v = np.sqrt(v2av_component) # the average x and y component of velocity
        # the new particles need to have same average KE
        # in this way it keeps the temperature fixed
        
        
        for i in range(n):
            r = grid[randompick2[i]]
            b = Ball(M[i], RB[i], r, [v[i], v[i]], Col[i], RBtot)
            self._blist.append(b)
            self._Rblist.append(RB[i])
        
        for i in range(self._N):
            self._blist[i]._rlist = RBtot #update for previous balls
        
        self._N = len (self._blist)
        
        self._container = Ball(100, self.__Rc, c = "None" , rl  = self._Rblist) 
        self._blist.append(self._container)
        self._Rblist.append(self.__Rc)
            
        self.randompick = self.randompick.tolist() + randompick2.tolist()
        self.randompick = np.array(self.randompick)
              
        self.dt_i = []  
        self.jcount = []
        self.ib = 0     
        self.jb = 0     
        self._Dmom = 0  
