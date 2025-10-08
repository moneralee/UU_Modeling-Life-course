import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from circuitsfunctions import *

########### Part I
def ODEgeneRegulation(a,t,parameters): 
    prod=parameters['prod']
    decay=parameters['decay']
    Ksat=parameters['Ksat']
    nodeA=parameters['nodeA']
    nodeB=parameters['nodeB']
    n=parameters['n']
    outputC=a[0]
    doutputC=prod*nodeA**n/(Ksat**n+nodeA**n)*nodeB**n/(Ksat**n+nodeB**n)-decay*outputC  #AND gate # modify this line for other logic gates
    return(doutputC)

def logicalRule(nodeA,nodeB):
    return(nodeA and nodeB)

# Main function to run the ODE model and Boolean model
def main():
    # ODE model - see the parameters used in circuitsfunctions.py
    ODErun(ODEgeneRegulation, 10, 10)

    # Boolean model
    # print output of function, look at the terminal for the result:
    print("the boolean operation of nodeA 1 AND nodeB 1 is:", logicalRule(1, 1))  # 1 AND 1 = 1

if __name__ == "__main__":
    main()