# Code created by Monica Garcia Gomez for Modeling Life course, Utrecht University
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import os
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# list of functions you will find here:
# You don't need to modify most of them to do the assignment, but you can check them to understand how they work. 
# 1 findNeighbors: finds the neighbors of a cell in the grid. Useful to update the transport of auxin
# 2 auxinTransport: updates the auxin grid based on the cell types and auxin transport parameters. 
# It considers the polarity of the auxin efflux transporters (PIN) in each cell type.
# 3 rootNetwork: defines the ODE system for the gene regulatory network in each cell. Same network as in our last practical! 
# This one you need to modify.
# 4 initialCondition: sets up the initial conditions for the simulation. Creates the cell grid and the 18 gene grids.
# 5 nodeUpdate: updates the gene expression levels in each cell after solving the ODEs. 
# 6 plotGrids: visualizes the cell grid and the auxin and gene expression grids in a single plot.

def findNeighbors(y,x,j,i): #i is compared with x, j with y
    neighbours = np.zeros((8, 2), dtype=int) #maximum 8 neighbours
    if(i==0): # cell in the left border
        if(j==0): # top
            neighbours[0]=(j,i+1)
            neighbours[1]=(j+1,i)
            neighbours[2]=(j+1,i+1)
            neighbours=neighbours[0:3]
        elif(j==y-1): # bottom
            neighbours[0]=(j-1,i)
            neighbours[1]=(j,i+1)
            neighbours[2]=(j-1,i+1)
            neighbours=neighbours[0:3]
        else:
            neighbours[0]=(j,i+1)
            neighbours[1]=(j+1,i+1)
            neighbours[2]=(j-1,i+1)
            neighbours[3]=(j-1,i)
            neighbours[4]=(j+1,i)
            neighbours=neighbours[0:5]       
    elif(i==x-1): # cell in the right border
        if(j==0): # top
            neighbours[0]=(j,i-1)
            neighbours[1]=(j+1,i)
            neighbours[2]=(j+1,i-1)
            neighbours=neighbours[0:3]
        elif(j==y-1): # bottom
            neighbours[0]=(j-1,i)
            neighbours[1]=(j,i-1)
            neighbours[2]=(j-1,i-1)
            neighbours=neighbours[0:3]
        else:
            neighbours[0]=(j+1,i)
            neighbours[1]=(j-1,i)
            neighbours[2]=(j+1,i-1)
            neighbours[3]=(j,i-1)
            neighbours[4]=(j-1,i-1)
            neighbours=neighbours[0:5]    
    elif(j==0): # cells in top border
        neighbours[0]=(j,i-1)
        neighbours[1]=(j,i+1)
        neighbours[2]=(j+1,i-1)
        neighbours[3]=(j+1,i)
        neighbours[4]=(j+1,i+1)
        neighbours=neighbours[0:5]
    elif(j==y-1): # cells in bottom border
        neighbours[0]=(j,i-1)
        neighbours[1]=(j,i+1)
        neighbours[2]=(j-1,i-1)
        neighbours[3]=(j-1,i)
        neighbours[4]=(j-1,i+1)
        neighbours=neighbours[0:5]     
    else: # rest, cells in the middle
        neighbours[0]=(j,i-1)
        neighbours[1]=(j,i+1)
        neighbours[2]=(j+1,i-1)
        neighbours[3]=(j+1,i)
        neighbours[4]=(j+1,i+1)
        neighbours[5]=(j-1,i-1)
        neighbours[6]=(j-1,i)
        neighbours[7]=(j-1,i+1) 
        neighbours=neighbours[0:8]           
    return neighbours

def auxinTransport(cellgrid,auxingrid,x,y,auxinSource,PAT,passiveTransport,auxinDegradation):
    auxingridn = auxingrid.copy() # new auxin grid to perform all updates and ensure mass conservation. Final exchange is performed at the end 
    #Auxin difussion 
    for i in range(x):
        for j in range(y):
            neighbours=findNeighbors(y,x,j,i) 
            totalneighbours=np.shape(neighbours)[0]
            auxinNeighbours=0
            for neigh in range(totalneighbours):
                auxinNeighbours+=auxingrid[neighbours[neigh,0], neighbours[neigh,1]] # how much auxin the neighbours have
                auxingridn[j,i]=auxingrid[j,i]+passiveTransport*(auxinNeighbours-totalneighbours*auxingrid[j,i]) # firs auxingridn update + how much auxin the cell receives and gives to neighbours
 #Auxin active transport (PIN-mediated)
    for i in range(x):
        for j in range(y):
            if cellgrid[j,i] == 6 or cellgrid[j,i] == 5: #Vascular or QC cells - basal PINs - down
                auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                auxingridn[j+1,i]=auxingridn[j+1,i]+PAT*(auxingrid[j,i])    
            elif cellgrid[j,i] ==2 or cellgrid[j,i] == 2.1: #epidermis - apical PINs - up
                auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*(auxingrid[j,i]) 
            elif cellgrid[j,i] ==4 or cellgrid[j,i] == 4.1: #endodermis - basal and inner-lateral PINs - down and side
                if i==2: #left epidermis
                    auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                    auxingridn[j+1,i]=auxingridn[j+1,i]+PAT*0.5*(auxingrid[j,i])    
                    auxingridn[j,i+1]=auxingridn[j,i+1]+PAT*0.5*(auxingrid[j,i])    
                elif i==x-3: #right epidermis
                    auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                    auxingridn[j+1,i]=auxingridn[j+1,i]+PAT*0.5*(auxingrid[j,i])    
                    auxingridn[j,i-1]=auxingridn[j,i-1]+PAT*0.5*(auxingrid[j,i])    
            elif cellgrid[j,i] ==3 or cellgrid[j,i] == 3.1: #cortex - apical and lateral PINs - up side
                if i==2: #left cortex
                    auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                    auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*0.5*(auxingrid[j,i])    
                    auxingridn[j,i+1]=auxingridn[j,i+1]+PAT*0.5*(auxingrid[j,i])    
                elif i==x-3: #right cortex
                    auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                    auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*0.5*(auxingrid[j,i])    
                    auxingridn[j,i-1]=auxingridn[j,i-1]+PAT*0.5*(auxingrid[j,i])    
            elif cellgrid[j,i] == 1: #columella - PINs in all sides
                if(i==0): #left border columella
                    if(j==(y-1)):
                        auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                        auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*(1/3)*(auxingrid[j,i])    
                        auxingridn[j,i+1]=auxingridn[j,i+1]+PAT*(1/3)*(auxingrid[j,i])    
                        auxingridn[j-1,i+1]=auxingridn[j-1,i+1]+PAT*(1/3)*(auxingrid[j,i])  
                    else:
                        auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                        auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j-1,i+1]=auxingridn[j-1,i+1]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j,i+1]=auxingridn[j,i+1]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j+1,i+1]=auxingridn[j+1,i+1]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j+1,i]=auxingridn[j+1,i]+PAT*(1/5)*(auxingrid[j,i])    
                elif(i==x-1): #right border columella
                    if(j==(y-1)):
                        auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                        auxingridn[j,i-1]=auxingridn[j,i-1]+PAT*(1/3)*(auxingrid[j,i])    
                        auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*(1/3)*(auxingrid[j,i])    
                        auxingridn[j-1,i-1]=auxingridn[j-1,i-1]+PAT*(1/3)*(auxingrid[j,i])   
                    else:
                        auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                        auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j-1,i-1]=auxingridn[j-1,i-1]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j,i-1]=auxingridn[j,i-1]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j+1,i-1]=auxingridn[j+1,i-1]+PAT*(1/5)*(auxingrid[j,i])    
                        auxingridn[j+1,i]=auxingridn[j+1,i]+PAT*(1/5)*(auxingrid[j,i]) 
                elif(j==(y-1)): #cells at bottom columella
                    auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                    auxingridn[j,i-1]=auxingridn[j,i-1]+PAT*(1/3)*(auxingrid[j,i])    
                    auxingridn[j,i+1]=auxingridn[j,i+1]+PAT*(1/3)*(auxingrid[j,i])    
                    auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*(1/3)*(auxingrid[j,i]) 
                else: # transport to all columella cells (middle)
                    auxingridn[j,i]=auxingridn[j,i]-PAT*(auxingrid[j,i]) 
                    auxingridn[j-1,i]=auxingridn[j-1,i]+PAT*(1/8)*(auxingrid[j,i])    
                    auxingridn[j-1,i+1]=auxingridn[j-1,i+1]+PAT*(1/8)*(auxingrid[j,i])    
                    auxingridn[j,i+1]=auxingridn[j,i+1]+PAT*(1/8)*(auxingrid[j,i])    
                    auxingridn[j+1,i+1]=auxingridn[j+1,i+1]+PAT*(1/8)*(auxingrid[j,i])    
                    auxingridn[j+1,i]=auxingridn[j+1,i]+PAT*(1/8)*(auxingrid[j,i])    
                    auxingridn[j+1,i-1]=auxingridn[j+1,i-1]+PAT*(1/8)*(auxingrid[j,i])    
                    auxingridn[j,i-1]=auxingridn[j,i-1]+PAT*(1/8)*(auxingrid[j,i])    
                    auxingridn[j-1,i-1]=auxingridn[j-1,i-1]+PAT*(1/8)*(auxingrid[j,i])    
    # Auxin degradation, influx in vascular cells 
    for i in range(x):
        for j in range(y):
            auxingridn[j,i]-=auxingridn[j,i]*auxinDegradation 
    auxingridn[0,3:x-3]+= auxinSource # only in vascular cells at the top
    return auxingridn

def rootNetwork(state,t,parameters):
    h = parameters['h']
    lambda_ = parameters['lambda']
    AUXIN = parameters['AUXIN']
    auxininput=parameters['auxininput']
    
    (CK, ARR1, SHY2, AUXIAAR, ARFR, ARF10, ARF5, XAL1, PLT, AUX, SCR, SHR, MIR165, PHB, JKD, MGP, WOX5, CLE40) = state
    
    # Node inputs
    w_CK = max(min(PHB, 1-ARFR), 1-SHR)
    w_ARR1 = min(CK, 1-SCR)
    w_SHY2 = min(1-AUX, ARR1)
    w_AUXIAAR = 1-AUX
    w_ARFR = 1-AUXIAAR
    w_ARF10 = min(1-AUXIAAR, 1-min(JKD, SHR))
    w_ARF5 = min(max(PHB, PLT), 1-min(SHR, MGP), 1-AUXIAAR, 1-SHY2)
    w_XAL1 = ARFR
    w_PLT = max(ARF5, XAL1, ARFR, WOX5)
    w_AUX = AUXIN
    w_SCR = min(SHR, SCR, JKD)
    w_SHR = SHR
    w_MIR165 = max(1-PHB, min(SCR, SHR, 1-ARR1))
    w_PHB = min(max(min(1-ARR1, PLT), PHB), 1-MIR165)
    w_JKD = min(1-PHB, SCR, SHR)
    w_MGP = min(1-ARF5, SCR, SHR)
    w_WOX5 = min(1-ARF10, ARF5, 1-CLE40)
    w_CLE40 = 1-SHR
    
    def calc_derivative(w, state_var):
        return ((-np.exp(0.5 * h) + np.exp(-h * w)) / ((1 - np.exp(0.5 * h)) * (1 + np.exp(-h * (w - 0.5))))) - (lambda_ * state_var)
    
    dCK = calc_derivative(w_CK, CK)
    dARR1 = calc_derivative(w_ARR1, ARR1)
    dSHY2 = calc_derivative(w_SHY2, SHY2)
    dAUXIAAR = calc_derivative(w_AUXIAAR, AUXIAAR)
    dARFR = calc_derivative(w_ARFR, ARFR)
    dARF10 = calc_derivative(w_ARF10, ARF10)
    dARF5 = calc_derivative(w_ARF5, ARF5)
    dXAL1 = calc_derivative(w_XAL1, XAL1)
    dPLT = calc_derivative(w_PLT, PLT)
    dAUX = calc_derivative(w_AUX, AUX)
    dSCR = calc_derivative(w_SCR, SCR)
    dSHR = calc_derivative(w_SHR, SHR)
    dMIR165 = calc_derivative(w_MIR165, MIR165)
    dPHB = calc_derivative(w_PHB, PHB)
    dJKD = calc_derivative(w_JKD, JKD)
    dCLE40 = calc_derivative(w_CLE40, CLE40)
    dMGP = calc_derivative(w_MGP, MGP)
    dWOX5 = calc_derivative(w_WOX5, WOX5)
# Solution to practical:
# Auxin regulates MGP. These has an indirect positive effect on ARF5. 
    dMGP = (45/(45+auxininput))*((-np.exp(0.5 * h) + np.exp(-h * w_MGP)) / ((1 - np.exp(0.5 * h)) * (1 + np.exp(-h * (w_MGP - 0.5))))) - (lambda_ * MGP)
# Auxin represses WOX5 at *very* high levels - most hormones exhibit this dosage dependent effect.
    dWOX5 = (1000/(1000+auxininput))*((-np.exp(0.5 * h) + np.exp(-h * w_WOX5)) / ((1 - np.exp(0.5 * h)) * (1 + np.exp(-h * (w_WOX5 - 0.5)))))-lambda_*WOX5  
    return [dCK, dARR1, dSHY2, dAUXIAAR, dARFR, dARF10, dARF5, dXAL1, dPLT, dAUX, dSCR, dSHR, dMIR165, dPHB, dJKD, dMGP, dWOX5, dCLE40]

def initialCondition(x,y):
    cellgrid = np.zeros((y, x), dtype=int) #first position in y height, second position in x width
    #cell types: 1 Col, 2 Epid, 3 Cortex, 4 Endodermis, 5 QC, 6 Vascular
    #making initial condition
    cellgrid[y-4:y,:]=1 #Columella
    cellgrid[y-5,0:x]=2.1 # Epid- bottom layer
    cellgrid[y-5,1:x-1]=3.1 # Cortex- SC layer
    cellgrid[y-5,2:x-2]=4.1 # Endodermis
    cellgrid[y-5,3:x-3]=5 # QC
    cellgrid[0:y-5,0:x]=2 # Epidermis
    cellgrid[0:y-5,1:x-1]=3 # Cortex
    cellgrid[0:y-5,2:x-2]=4 # Endodermis
    cellgrid[0:y-5,3:x-3]=6 # Vascular
    ckgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    arr1grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    shy2grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    auxiaagrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    arfrgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    arf10grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    arf5grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    xal1grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    pltgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    auxgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    scrgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    shrgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    mir165grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    phbgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    jkdgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    mgpgrid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    wox5grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    cle40grid = np.zeros((y, x), dtype=float) #first position in y height, second position in x width
    for i in range(x): # initializing all genes/hormones in their experimentally determined expression/activity patterns
        for j in range(y):
            if cellgrid[j,i] == 1:   #Columella
                ckgrid[j,i] = 1
                arr1grid[j,i] = 1
                arfrgrid[j,i] = 1
                arf10grid[j,i] = 1
                arf5grid[j,i] = 1
                xal1grid[j,i] = 1
                pltgrid[j,i] = 1
                auxgrid[j,i] = 1
                cle40grid[j,i] = 1
            elif cellgrid[j,i] == 4 or cellgrid[j,i] == 4.1: #endodermis
                arfrgrid[j,i] = 1
                xal1grid[j,i] = 1
                pltgrid[j,i] = 1
                auxgrid[j,i] = 1
                scrgrid[j,i] = 1
                shrgrid[j,i] = 1
                mir165grid[j,i] = 1
                jkdgrid[j,i] = 1
                mgpgrid[j,i] = 1
            elif cellgrid[j,i] == 5: # QC
                arfrgrid[j,i] = 1
                arf5grid[j,i] = 1
                xal1grid[j,i] = 1
                pltgrid[j,i] = 1
                auxgrid[j,i] = 1
                scrgrid[j,i] = 1
                shrgrid[j,i] = 1
                mir165grid[j,i] = 1
                jkdgrid[j,i] = 1
                wox5grid[j,i] = 1
            elif cellgrid[j,i] == 6: #Vascular
                arfrgrid[j,i] = 1
                arf10grid[j,i] = 1
                arf5grid[j,i] = 1
                xal1grid[j,i] = 1
                pltgrid[j,i] = 1
                auxgrid[j,i] = 1
                shrgrid[j,i] = 1
                phbgrid[j,i] = 1

    return cellgrid,ckgrid, arr1grid, shy2grid, auxiaagrid, arfrgrid, arf10grid, arf5grid, xal1grid, pltgrid, auxgrid, scrgrid,shrgrid,mir165grid,phbgrid,jkdgrid,mgpgrid, wox5grid,cle40grid

def nodeUpdate(result,ck,arr1,shy2,auxiaa,arfr,arf10,arf5,xal1,plt,aux,scr,shr,mir165,phb,jkd,mgp,wox5,cle40):
    # saving ODE results into grids
    ck=result[-1,0]
    arr1 = result[-1,1]
    shy2 = result[-1,2]
    auxiaa = result[-1,3]
    arfr = result[-1,4]
    arf10 = result[-1,5]
    arf5 = result[-1,6]
    xal1 = result[-1,7]
    plt = result[-1,8]
    aux = result[-1,9]
    scr = result[-1,10]
    shr = result[-1,11]
    mir165 = result[-1,12]
    phb = result[-1,13]
    jkd = result[-1,14]
    mgp = result[-1,15]
    wox5 = result[-1,16]
    cle40 = result[-1,17]
    return ck, arr1, shy2, auxiaa, arfr, arf10, arf5, xal1, plt, aux, scr, shr, mir165, phb, jkd, mgp, wox5, cle40

def plotGrids(cellgrid,auxingrid,arf10grid,arf5grid,mgpgrid,wox5grid,auxinTreatment,t):
    plt.suptitle(f"Auxin treatment: {auxinTreatment}")
    plt.subplot(1, 6, 1)
    plt.imshow(cellgrid, cmap='viridis', vmin=0, vmax=6)
    plt.title('Cells')
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.imshow(np.log1p(auxingrid * (1 + auxinTreatment)), cmap='plasma', vmin=0, vmax=np.log1p(100)) #in linear map i used vmax 100
    plt.title('Auxin')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(arf10grid, cmap='Purples',vmin=0, vmax=1)
    plt.title('ARF10')
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.imshow(arf5grid, cmap='Purples',vmin=0, vmax=1)
    plt.title('ARF5')
    plt.axis('off')
    plt.subplot(1, 6, 5)
    plt.imshow(mgpgrid, cmap='Purples',vmin=0, vmax=1)
    plt.title('MGP')
    plt.axis('off')
    plt.subplot(1, 6, 6)
    plt.imshow(wox5grid, cmap='Purples',vmin=0, vmax=1)
    plt.title('WOX5')
    plt.axis('off')
    plt.pause(0.1) #make it slower if it doesn't show roots
    plt.savefig(f'output/final_grids_run{auxinTreatment}_{t}.png', bbox_inches='tight', dpi=300)

