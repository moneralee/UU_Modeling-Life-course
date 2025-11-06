import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from Rootfunctions import *

# Grid properties
x = 8 # number of cell layers in radial direction 
y = 50 #number of cell layers

# Auxin transport parameters
auxinTreatment=0 # 0 for control, 10 or 750 for low and high auxin treatments
simulationSteps=100 
networkUpdate=simulationSteps/10 # network update is every 10 steps
auxinSource=0.1 # auxin influx from the vascular cells
PAT= 0.05
passiveTransport=PAT/10
auxinDegradation=passiveTransport/100 

def main():
    plt.ion() 
  
    #Initialize root grid and gene expression
    cellgrid,ckgrid, arr1grid, shy2grid, auxiaagrid, arfrgrid, arf10grid, arf5grid, xal1grid, pltgrid, auxgrid, \
        scrgrid,shrgrid,mir165grid,phbgrid,jkdgrid,mgpgrid, wox5grid,cle40grid = initialCondition(x,y) 
    auxingrid=np.load('auxin_grid.npy')
    plotGrids(cellgrid, auxingrid, arf10grid, arf5grid, mgpgrid, wox5grid, auxinTreatment, "IC")

    # Run simulation 
    for step in range(simulationSteps):
        # update auxin transport every timestep
        auxingrid=auxinTransport(cellgrid,auxingrid, x, y, auxinSource, PAT, passiveTransport,auxinDegradation)
        
        # update network every 10 steps
        if step % (networkUpdate) == 0: 
            for i in range(x): # each cell has a network, so we loop over all x and y gridpoints
                for j in range(y):
                    steps = np.arange(0, 2.1, 0.1)
                    parameters = {'h': 50, 'lambda': 1, 'AUXIN':1, 'auxininput': auxingrid[j,i]*(1+auxinTreatment)} # auxin feeds into the network here
                    # get current state of the cell 
                    cellState=[ckgrid[j,i], arr1grid[j,i],shy2grid[j,i],auxiaagrid[j,i],arfrgrid[j,i],
                                arf10grid[j,i],arf5grid[j,i],xal1grid[j,i],pltgrid[j,i],auxgrid[j,i],scrgrid[j,i],
                                shrgrid[j,i],mir165grid[j,i],phbgrid[j,i],jkdgrid[j,i],mgpgrid[j,i],wox5grid[j,i],
                                cle40grid[j,i]] # this combines all 18 activity values in a vector
                    
                    # ODE solver - takes as input the network model, current state, time steps and parameters
                    result = odeint(rootNetwork, cellState, steps, args=(parameters,)) # we solve the ODEs

                    ckgrid[j,i], arr1grid[j,i],shy2grid[j,i],auxiaagrid[j,i],arfrgrid[j,i],arf10grid[j,i],arf5grid[j,i],\
                        xal1grid[j,i],pltgrid[j,i],auxgrid[j,i],scrgrid[j,i],shrgrid[j,i],mir165grid[j,i],phbgrid[j,i],\
                            jkdgrid[j,i],mgpgrid[j,i],wox5grid[j,i],cle40grid[j,i]=nodeUpdate(result,ckgrid[j,i], arr1grid[j,i],\
                                shy2grid[j,i],auxiaagrid[j,i],arfrgrid[j,i],arf10grid[j,i],arf5grid[j,i],xal1grid[j,i],pltgrid[j,i],\
                                    auxgrid[j,i],scrgrid[j,i],shrgrid[j,i],mir165grid[j,i],phbgrid[j,i],jkdgrid[j,i],mgpgrid[j,i],\
                                        wox5grid[j,i],cle40grid[j,i]) # update gene values in the gridpoint
        # plot every 10 steps
        if step % 10 == 0:
            plotGrids(cellgrid, auxingrid, arf10grid, arf5grid, mgpgrid, wox5grid, auxinTreatment, step)

    #Save final grids
    plotGrids(cellgrid,auxingrid,arf10grid,arf5grid,mgpgrid,wox5grid,auxinTreatment,simulationSteps)


if __name__ == "__main__":
    main()