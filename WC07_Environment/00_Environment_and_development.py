import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""Parameters""" 
Pmax = 1     #Max Photosynthesis rate
R25  = 0.2   #Respiration rate at 25C
Gmax = 0.25  #Max growth rate
K_g  = 2     #Value where growth is at half max.

#Three temperatures to test
T1=15; T2=25;T3=35
#Two angles to test
angle1 = 20; angle2 = 40

## Calculate Respiration (1/m2/s) normalized for a reference temperature of 25C
def Respiration_per_m2(T,Q10):
    Resp    = Q10 ** ( (T-25) / 10) 

    return Resp

## Calculate Realtive Photosynthesis with a reference temperetaure of 25C
def Photosynthesis_per_m2(T):
    # This function calculates the temperature dependency of photosynthesis parameter values  
    def TempDependency(c, dHa,T):
        R = 8.31e-3
        T = T + 273
        return  np.exp(c - dHa/(R*T))

    #calculate photosynthesis at temperature T and at 25C and returns A_T/A_25
    ci = 230; O = 210
    VcMax  = 98*TempDependency(26.35,65.33,T) 
    VoMax  = 98*TempDependency(22.98,60.11,T) 
    Kc     = TempDependency(38.05,79.43,T) 
    Ko     = TempDependency(20.30,36.38,T) 
    G_Star = TempDependency(19.02,37.83,T) 
    R_D    = 1.1*TempDependency(18.72,46.39,T)   
    
    VcMax25  = 98*TempDependency(26.35,65.33,25) 
    VoMax25  = 98*TempDependency(22.98,60.11,25) 
    Kc25     = TempDependency(38.05,79.43,25) 
    Ko25     = TempDependency(20.30,36.38,25) 
    G_Star25 = TempDependency(19.02,37.83,25) 
    R_D25    = 1.1*TempDependency(18.72,46.39,25)           

    A = (1 - G_Star/ci) *( VcMax * ci / (ci + Kc*(1 + O/Ko))) - R_D #Photosynthesis Rate
    A25 = (1 - G_Star25/ci) *( VcMax25 * ci / (ci + Kc25*(1 + O/Ko25))) - R_D25 #Photosynthesis Rate at 25C
   

    return A/A25

def EffectiveLeafArea(LA,angle):

    #Calculate the effective leaf area based on the leaf area and angle.

    return LA

def Photosynthesis(LA, T,angle):

    # Photosynthesis rate depends on leaf Temperature
    TempPhoto = Photosynthesis_per_m2(T)

    #Photosynthesis depends on leaf angle - assuming light comes from the top
    LA_eff = EffectiveLeafArea(LA, angle)
    
    return Pmax * TempPhoto *  LA_eff
    
def Respiration(LA,T,Q10):
    
    TempResp    = R25 * Respiration_per_m2(T, Q10)
    return TempResp *  LA

    
def Growth(C_conc, LA):
    #LA increase saturates with Carbon concentration in the leaf
    return Gmax * C_conc/(C_conc+K_g) * LA 
    
def ODE_system(t, y, T, angle):
    C, LA = y;    C_conc=C/LA

    dCdt =  Photosynthesis(LA, T,angle) - Respiration(LA,T,2.5) - Growth(C_conc,LA)
    dLAdt = Growth(C_conc,LA)

    return [dCdt, dLAdt]

"""Investigate how photosynthesis (specificity vs. turnover) and respiration (Q10 principle, assume Q10 =2) depend on Temperature - use functions in HelperFunctions.py
Photosynthesis can be normalized to 25C by dividing A over A25 """



"""Run simulations of the ODE model for low (15°C), medium (25°C), and high (35°C) temperatures. 
Which plant performs best? Which plant worst? Why is this?"""

# Initial conditions
y0 = [2.5, 2.5]  #Initial LA and C state of the plant
# Time span of the simulation
tend=24; t_eval = np.linspace(0, tend, 100)
#Simulation for T1=15    
T15 = solve_ivp(ODE_system, (0,tend), y0, t_eval=t_eval, args=(T1,25,))
# Plot results
fig,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(T15.t, T15.y[1],label='T=15')
ax1.set_xlabel('Time (-)'); ax1.set_ylabel('Leaf Area')
ax1.legend()
ax1.set_ylim(0,50)

ax2.plot(T15.t, T15.y[0]/T15.y[1],label='T=15')
ax2.set_xlabel('Time (-)'); ax2.set_ylabel('Carbon')
ax2.legend()
plt.tight_layout()

plt.savefig('Growth_Q1.png')

plt.show()
print(T15)


""""
Until now, we have looked at the differences between temperatures for plants that do not respond to their environment
In reality, plants respond to their surroundings. Plants have a so-called thermomorpogenic response, in which it is observed that they lift
their leaves (leaf hyponasty, increased leaf angle) when temperatures are high. This is thought to be a response to prevent overheating of the leaves.
"""
#plot effective leaf area as a function of angle




#Use a barplot to plot the final LA size for comparison


""""Thee thermomorphogenesis repsonse does also have a positive effect. It is shown to lower leaf temperature, 
which increases photosynthesis and decreases respiration.Investigate if this can have an effect, 
i.e. how big is the effect for 3/5/7 degrees cooling down?"""

#Make a barplot for the different angles and temperatures
#Or/and, plot final leaf area as a function of temperature (between 20-40C for two angles)


