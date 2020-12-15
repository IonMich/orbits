#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:24:33 2018

@author: yannis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random

M_sun = 1.0             ## Mass of Earth's Sun in Solar Masses
G = 0.000296005155      ## Gravitational Constant in AU^3 per Solar Masses per Days^2
M_jupiter = 1/1047.35 * M_sun

class astroObject:    
    """
    Create an astrophysical object with defined mass 2-D position and 2-D velocity
    """
    def __init__(self, mass, initX, initY, initVelX, initVelY):
        self.M = mass
        self.x = initX
        self.y = initY
        self.velX = initVelX
        self.velY = initVelY
        
    def distFromObj(self,otherObj):
        """
        Calculate the distance of this object  from another object
        
        Return the distance as a float 
        """
        return ( ( self.x - otherObj.x )**2 + ( self.y - otherObj.y )**2 ) ** 0.5
    
    def accFromOthers(self, aListOfObjects):
        """
        Calculate the acceleration vector components of this object given a set of gravitational objects        
        
        Return a tuple of the x- and y-components of the acceleration
        """        
        accX = 0
        accY = 0
        for otherObj in aListOfObjects:
            if otherObj != self:
                acc = G * otherObj.M / self.distFromObj(otherObj)**3
                accX += acc * (otherObj.x - self.x)
                accY += acc * (otherObj.y - self.y)
        return accX, accY
    
    def copy(self, newArray=(None,)*4):
        """
        Copy the object to a new astroObject and assign to it new position and velocity as a tuple of length 4
        
        If new configuration is not specified, keep previous pos and vel
        """
        if newArray.all() == None:
            x , y , velx , vely = self.x, self.y, self.velX, self.velY
        else:
            x , y , velx , vely = newArray
        return astroObject(self.M,  x, y,velx,vely)

def initPlanet(planetPeriod):
    """
    Randomize the initial position of the planet
    """
    planetDistance = ( planetPeriod**2 / (2*np.pi)**2 * G * (mSum +mPlanet) )**(1/3)
    orbitRadiusPlanet = mSum/(mSum + mPlanet) * planetDistance
    initVelPlanet = 2 * np.pi * orbitRadiusPlanet / planetPeriod
    thePlanet = astroObject(mPlanet, 0,orbitRadiusPlanet, initVelPlanet,0)
        
    # randomize the exact initial position of the planet
    theta = random.uniform(0,2*np.pi)
    thePlanet.x , thePlanet.y = ( np.sin(theta) , np.cos(theta) ) * np.array(thePlanet.y)
    thePlanet.velX , thePlanet.velY = ( np.cos(theta) , -np.sin(theta) ) * np.array(thePlanet.velX)
    
    return thePlanet
    
def fixMomenta(thePlanet,star1,star2):
    """
    Subtract momentum of the planet from the system
    """
    ## Subtract the momentum of the planet from the system
    ## so that there is no overall drift in that direction
    star1.velX , star1.velY = (0 , initVel1) - 0.5 * (thePlanet.M / star1.M) * np.array([thePlanet.velX,thePlanet.velY])
    star2.velX , star2.velY = (0 , -initVel2) - 0.5 * (thePlanet.M / star2.M) * np.array([thePlanet.velX,thePlanet.velY])


mStar1 = 0.68 * M_sun
mStar2 = 0.20 * M_sun
mSum = mStar1 + mStar2
mPlanet = 0.33 * M_jupiter
#mPlanet = 0  ## Uncomment to remove influence of the planet to the stars

pStar = 41              ## (Initial) period of star orbits in days

starsDistance = ( pStar**2 / (2*np.pi)**2 * G * (mSum) )**(1/3)

orbitRadius1 = mStar2/(mSum) * starsDistance
orbitRadius2 = mStar1/(mSum) * starsDistance

initVel1 = 2 * np.pi * orbitRadius1 / pStar
initVel2 = 2 * np.pi * orbitRadius2 / pStar

## Create astrophysical objects
star1 = astroObject(mStar1, orbitRadius1,0, 0,initVel1)
star2 = astroObject(mStar2, -orbitRadius2,0, 0,-initVel2)
      


def adaptStep(h,ri,method,lambdaFunc,sympl=False):
    """
    Adapt the step size to reduce local error to some specified relative error (in our case 1E-3) 
    """
    riTmp = ri.copy()
    riTmp2 = ri.copy()
    
    ## Evolve with two steps of h
    riTmp = method(lambdaFunc,riTmp,h)    
    riTmp = method(lambdaFunc,riTmp,h)
    
    ##Evolve with one step of 2*h
    riTmp2 = method(lambdaFunc,riTmp2,2*h)
    
    ## Compare
    if sympl:
        maxRelDist = max(np.sqrt((riTmp[0]-riTmp2[0])**2 + (riTmp[1]-riTmp2[1])**2) / np.sqrt( (riTmp[0])**2 + (riTmp[1])**2),
                         np.sqrt((riTmp[2]-riTmp2[2])**2 + (riTmp[3]-riTmp2[3])**2) / np.sqrt( (riTmp[2])**2 + (riTmp[3])**2),
                         np.sqrt((riTmp[4]-riTmp2[4])**2 + (riTmp[5]-riTmp2[5])**2) / np.sqrt( (riTmp[4])**2 + (riTmp[5])**2),
                         ## Uncomment for second planet
#                         np.sqrt((riTmp[6]-riTmp2[6])**2 + (riTmp[7]-riTmp2[7])**2) / np.sqrt( (riTmp[6])**2 + (riTmp[7])**2) 
                         )
    else:
        maxRelDist = max(np.sqrt((riTmp[0]-riTmp2[0])**2 + (riTmp[1]-riTmp2[1])**2) / np.sqrt( (riTmp[0])**2 + (riTmp[1])**2),
                         np.sqrt((riTmp[4]-riTmp2[4])**2 + (riTmp[5]-riTmp2[5])**2) /  np.sqrt( (riTmp[4])**2 + (riTmp[5])**2),
                         np.sqrt((riTmp[8]-riTmp2[8])**2 + (riTmp[9]-riTmp2[9])**2) / np.sqrt( (riTmp[8])**2 + (riTmp[9])**2),
#                         np.sqrt((riTmp[12]-riTmp2[12])**2 + (riTmp[13]-riTmp2[13])**2) / np.sqrt( (riTmp[12])**2 + (riTmp[13])**2) 
                         )

    
    h = h * (relError / maxRelDist )**(1/5)
    
    return h

def rk4(lFunc, myRi,h):
    """
    Implement RK4 to evolve myRi by h
    """
    k1 = h*lFunc(myRi)
    k2 = h*lFunc(myRi+0.5*k1)
    k3 = h*lFunc(myRi+0.5*k2)
    k4 = h*lFunc(myRi+k3)
    myRi += (k1 + 2*k2 + 2*k3 + k4)/6
    return myRi

def symplectic2(lFunc, myRi,h):
    """
    Implement the Verlet method to evolve myRi by h
    """
    velHalf = myRi[2*numObjs:] + 0.5 * lFunc(myRi) * h
    newPos = myRi[:2*numObjs] + velHalf * h
    newAcc = lFunc( np.concatenate((newPos,velHalf)) )
    newVel = velHalf + 0.5 * newAcc * h
    
    return np.concatenate((newPos,newVel))

def symplectic4(lFunc, myRi,h):
    """
    Implement the symplected integrator of 4th order to evolve myRi by h
    Coefficients found in: 
    http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-5071.pdf
    """
    x = (2**(1/3) + 2**(-1/3) - 1)/6
    
    ## TODO: Find what is wrong with this choice
#    c1 = c4 = x + 1/2
#    c2 = c3 = -x
#    d1 = d3 = 2*x + 1
#    d2 = - 4*x - 1
    
    c1 = 0
    c3 = - 4*x - 1
    c2 = c4 = 2*x + 1
    d2 = d3 = -x
    d1 = d4 = x + 1/2
    
    vel = myRi[2*numObjs:] + c1 * lFunc(myRi) * h
    pos = myRi[:2*numObjs] + d1 * vel * h
    acc = lFunc( np.concatenate((pos,vel)) )
    vel += c2 * acc * h
    pos += d2 * vel * h
    acc = lFunc( np.concatenate((pos,vel)) )
    vel += c3 * acc * h
    pos += d3 * vel * h
    acc = lFunc( np.concatenate((pos,vel)) )
    vel += c4 * acc * h
    pos += d4 * vel * h
    
    return np.concatenate((pos,vel))
    
    
  
def timeEvolution(method,lambdaFunc,t0,t1,myInitList,hInit, adaptive=True, symplectic=False):
    """Evolve the astrophysical system with apaptive time-step
    from time t0 to time t1 with an initial timestep of hInit
    Take as an argument the integration method (RK45, symplectic2 symplectic4,...)
    Select the appropriate function for lambfaFunc (f2 for RK , fSymplectic for symplectic integrators)
    Select between adaptive and fixed time-stepping
    Update the objects with their final configuration
    Return: t and r and h (both np.ndarray objects) and also  boolean of whether the planet escaped
    """
    t = t0
    tList = np.array([t])
    
    ri = np.array(myInitList,float)
    r = np.array([ri])
    
    h = hInit
    hList = np.array([h])
    
    escaped = False
    ii = 0
    while t < t1:
        ## adapt timestep
        if adaptive :
            h = adaptStep(h,ri,method,lambdaFunc,sympl=symplectic)
            
        if ii%10000 == 0:
                print("Current epoch: {:.2f} years\nCurrent time-step: {:.2f} days".format(t/365,h))
        
        ri = method(lambdaFunc,ri,h)

        t += h
        ii += 1
        
        tList = np.append(tList,np.array([t]))
        r = np.append(r,np.array([ri]),axis=0)
        hList = np.append(hList,np.array([h]))
        if ( np.abs(ri[4]) + np.abs(ri[5])>10 and symplectic) or (np.abs(ri[8])+np.abs(ri[9])>10 and (not symplectic) ): 
            escaped = True
            print("Planet: 'See ya!'")
            break
    # update objects
    for i , astroObj in enumerate( listOfObjects ):
        if symplectic:
            listOfObjects[i] = astroObj.copy(ri[ np.array([2*i,2*i+1,2*numObjs+2*i,2*numObjs+2*i+1]) ])
        else:
            listOfObjects[i] = astroObj.copy(ri[ 4*i : 4*i+4 ])

    return tList,r , hList, escaped


def f2(perturbedArray):
    """
    Calculate the derivatives of the phase space coordinates of astrophysical 
    objects in 2-D. 
    Uses astroObject objects to calculate the accelaration 
    
    Return a tuple of length 4*(number of objects) with the derivatives
    in a format that is more appropriate for RK integration
    """
    farray = np.empty(4*len(listOfObjects))
    
    ## update temprarily objects
    newObjects = []
    for i , astroObj in enumerate( listOfObjects ):
        newObj = astroObj.copy(perturbedArray[ 4*i : 4*i+4 ])
        newObjects.append(newObj)
    
    ## Calculate the derivatives of the phase space coordinates
    for i , astroObj in enumerate( newObjects ):
        accX , accY = astroObj.accFromOthers(newObjects)
        farray[4*i:4*i+4] = astroObj.velX , astroObj.velY , accX, accY
    
    return farray
   
def fSymplectic(perturbedArray):
    """
    Calculate the derivatives of the phase space coordinates of astrophysical 
    objects in 2-D. 
    Uses astroObject objects to calculate the accelaration 
    

    Return a tuple of length 2*(number of objects) with the accelerations
    in a format that is more appropriate for symplectic integration
    """
    farray = np.empty(2*len(listOfObjects))
    
    ## update temprarily objects
    newObjects = []
    for i , astroObj in enumerate( listOfObjects ):
        newObj = astroObj.copy( perturbedArray[ np.array([2*i,2*i+1,2*numObjs+2*i,2*numObjs+2*i+1]) ] )
        newObjects.append(newObj)
    
    ## Calculate the derivatives of the phase space coordinates
    for i , astroObj in enumerate( newObjects ):
        accX , accY = astroObj.accFromOthers(newObjects)
        farray[np.array([2*i,2*i+1])] = accX, accY
    
    return farray     
        
def configArray(symplectic=False):
    """
    Return the configuration array of the phase space coordinates 
    for the list of astroObject instances
    """
    configArray = np.empty(4*len(listOfObjects))
    
    for i , astroObj in enumerate( listOfObjects ):
        if symplectic:
            configArray[np.array([2*i,2*i+1,2*numObjs+2*i,2*numObjs+2*i+1])] = astroObj.x , astroObj.y ,astroObj.velX , astroObj.velY
        else:
            configArray[4*i:4*i+4] = astroObj.x , astroObj.y , astroObj.velX , astroObj.velY
        
    return configArray
    

if __name__ == "__main__":
    periodList = [229,150,90,70,50,40] ## (Initial) periods of planet orbit in days
#    periodList = [229,150,] ## (Initial) periods of planet orbit in days
#    periodList =[100]
    
    t0 , t1 = 0 , 365 * 100
    relError = 1E-3
    wiThSymplectic = True
    
    fig = plt.figure(1,figsize=(12,6))
    fig.suptitle("Relative Error at each step: {}".format(relError))
    for index , pPlanet in enumerate(periodList):
        print("{} out of {}: Starting planet with period {:d} days".format(index+1,len(periodList),pPlanet))
        planet = initPlanet(pPlanet)
        fixMomenta(planet,star1,star2)
        listOfObjects = [star1,star2,planet]
        numObjs = len(listOfObjects)
        
#        ## add 2nd planet (total momentum will not be zero anymore)
#        pPlanet2 = 500
#        planet2 = initPlanet(pPlanet2)
#        fixMomenta(planet2,star1,star2)
#        listOfObjects = [star1,star2,planet,planet2]
#        numObjs = len(listOfObjects)        

        if wiThSymplectic:
            
            myMethod = symplectic4
            myFunc = fSymplectic
            initConfigArray = configArray(symplectic=wiThSymplectic)
        else:
            ## Runge Kutta 4-5
            myMethod = rk4
            myFunc = f2
            initConfigArray = configArray(symplectic=wiThSymplectic)
        
        hStep = 1
        t, r, h, escaped = timeEvolution(myMethod,myFunc,t0,t1,initConfigArray,hStep,symplectic=wiThSymplectic)
    
        ## RK4 indexing
        x1, y1, x2, y2, x3, y3= r[:,0], r[:,1], r[:,4], r[:,5], r[:,8],r[:,9] 
#        x4 ,y4 = r[:,12], r[:,13]
        
        if wiThSymplectic:
            ## symplectic indexing
            x2 , y2 , x3 , y3 = r[:,2] , r[:,3] , r[:,4] , r[:,5]
#            x4 ,y4 = r[:,6], r[:,7]
            
        r1 = np.sqrt(np.array(x1)**2 + np.array(y1)**2)
        r2 = np.sqrt(np.array(x2)**2 + np.array(y2)**2)
        rPl = np.sqrt(np.array(x3)**2 + np.array(y3)**2)
        
        #### Plotting
        
        ax = fig.add_subplot(2, 3, index+1)
        plt.plot([0,],[0,],"k.", MarkerSize=0.5)
        
        plot1 = plt.plot(x1,y1, color='blue',label="Star 1")
        plot2 = plt.plot(x2,y2, color='red', label="Star 2")
        plotPl = plt.plot(x3,y3, color='green',label="Planet")
#        plt.plot(x4,y4)
        
        if escaped:
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
        
        marker1 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                              markersize=10, label='Star 1')
        marker2 = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                  markersize=10, label='Star 2')
        markerPl = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                                  markersize=10, label='Planet')
        
        plt.legend(loc="lower left",handles=[marker1, marker2, markerPl])
        ax.set_title('Initial Planet period: {:d} days'.format(pPlanet))
        ax.set_aspect('equal')
        ax.set_xlabel('AU')
        ax.set_ylabel('AU')
        

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
        
print("\n Our program uses Symplectic Integration  of 4th order instead of Runge",
      "Kutta 4-5, in order to make sure that we preserve energy at each step.",
      "(Can be changed by setting wiThSymplectic=False)\n",
      "The program works with arbitrary number of planet (uncomment the relevant lines for a second planet).\n",
      "When we run the program for 100 years we see that the system becomes",
      "unstable when the initial period of the planet reaches ~60 days or a radius of ~0.3 AU.\n",
      "When we run our code for hundrends of years and lower relative error",
      "we see the minimum stable radius is probably much higher."                              
      ) 
    
#    plt.figure(2)
#    plt.plot(t/365,rPl)
#    
#    plt.figure(3)
#    plt.plot(t/365,r1)
#    
#    plt.figure(4)
#    plt.plot(t/365,h)
    