import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

    #This function calculates photosynthesis at temperature T and at 25C and returns A_T/A_25
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
    """
    Calculate the effective leaf area based on the leaf area and angle.
    """
    return np.cos(np.radians(angle)) * LA

def Photosynthesis(LA, T,angle,open):

    # Photosynthesis rate depends on leaf Temperature
    TempPhoto = Photosynthesis_per_m2(T)

    #Photosynthesis goes down with increased leaf angle - assuming light comes from the top
    LA_eff = EffectiveLeafArea(LA, angle)
    
    #Max Photosynthesis rate
    Pmax = 1
    
    return Pmax * TempPhoto *  LA_eff * open
    
def Respiration(LA,T,Q10):
    #Respiration rate at 25C
    R25 = 0.2 

    TempResp    = R25 * Respiration_per_m2(T, Q10)
    return TempResp *  LA

def Growth(C_conc, LA):
    #LA increase saturates with Carbon concentration in the leaf
    return 0.25 * C_conc/(C_conc+2) * LA 

def stomatal_opening(T):
    # Gaussian function parameters
    a = 2.173      # amplitude
    mu = 22#23.956    # optimum temperature (°C)
    sigma = 6.929  # spread (°C)
    c = 1.814      # baseline aperture (µm)
    return (a * np.exp(-0.5 * ((T - mu) / sigma)**2) + c)/4.0


def leaf_temperature(T, open, angle):
    # Leaf temperature decreases with stomatal opening
    # and decreases with leaf angle due to less direct sunlight
    #T_leaf = T  -(open*10) - (1-np.cos(np.radians(angle))) * 7.5  # arbitrary scaling factors
    T_leaf = T*((1- 0.175*open)*(1 - 0.3*(1-np.cos(np.radians(angle)))))
    return T_leaf


def ODE_system(t, y, T, angle):
    C, LA, Tleaf = y;    C_conc=C/LA

    T_eff = Tleaf
    open = stomatal_opening(T_eff)
 
    dCdt =  Photosynthesis(LA, T_eff,angle,open) - Respiration(LA,T_eff,2.5) - Growth(C_conc,LA)
    dLAdt = Growth(C_conc,LA)
    tau = 10  # Time constant for leaf temperature adjustment
    #alpha=(1- 0.29*open)*(1 - 0.3*(1-np.cos(np.radians(angle))))
    alpha=(1- 0.2*open)*(1 - 0.4*(1-np.cos(np.radians(angle)))) 
    dTleafdt = tau *(T*alpha -Tleaf)
    return [dCdt, dLAdt, dTleafdt]

""""Thee thermomorphogenesis repsonse does also have a positive effect. It is shown to lower leaf temperature, 
which increases photosynthesis and decreases respiration.Investigate if this can have an effect, 
i.e. how big is the effect for 3/5/7 degrees cooling down?"""

temps = np.linspace(10, 35, 100)
apertures = np.array([stomatal_opening(T) for T in temps])
plt.figure(figsize=(8, 5))
plt.plot(temps, apertures)
# Find and print the maximum aperture value
max_aperture = np.max(apertures)
print("Maximum aperture value:", max_aperture)

# Generate data for leaf temperature as a function of stomatal aperture and leaf angle
angles = np.linspace(0, 90, 100)  # Leaf angles from 0 to 90 degrees
apertures = np.linspace(0, 1, 100)  # Stomatal apertures from 0 to 1
# Create a meshgrid for angles and apertures
angle_grid, aperture_grid = np.meshgrid(angles, apertures)
# Calculate leaf temperature for each combination of angle and aperture
T_air = 30  # Ambient temperature in °C
leaf_temp_grid = np.array([
    [leaf_temperature(T_air, aperture, angle) for angle in angles]
    for aperture in apertures
])
# Plot the results
plt.figure(figsize=(8, 6))
contour = plt.contourf(angle_grid, aperture_grid, leaf_temp_grid, cmap='viridis', levels=20)
plt.colorbar(contour, label='Leaf Temperature (°C)')
plt.xlabel('Leaf Angle (degrees)')
plt.ylabel('Stomatal Aperture')
plt.title('Leaf Temperature as a Function of Stomatal Aperture and Leaf Angle')
plt.tight_layout()
plt.show()



#Two temperatures to test
T1=15;T2=25;T3=35
#Two angles to test
angle1 = 20; angle2 = 40
#Investigate the effect of leaf hyponasty
y0_1 = [2.5, 2.5,T1]  #Initial LA and C state of the plant
y0_2 = [2.5, 2.5,T2]  #Initial LA and C state of the plant
y0_3 = [2.5, 2.5,T3]  #Initial LA and C state of the plant
# Time span of the simulation
tend=24; t_eval = np.linspace(0, tend, 100)


# Collect final leaf area values for specified temperatures and angles
sol20_T25 = solve_ivp(ODE_system, (0, tend), y0_2, t_eval=t_eval, args=(25, 20,))
sol20_T35 = solve_ivp(ODE_system, (0, tend), y0_3, t_eval=t_eval, args=(35, 20,))
sol40_T25 = solve_ivp(ODE_system, (0, tend), y0_2, t_eval=t_eval, args=(25, 40,))
sol40_T35 = solve_ivp(ODE_system, (0, tend), y0_3, t_eval=t_eval, args=(35, 40,))
sol40_T34 = solve_ivp(ODE_system, (0, tend), y0_3, t_eval=t_eval, args=(34, 40,))
sol40_T33 = solve_ivp(ODE_system, (0, tend), y0_3, t_eval=t_eval, args=(33, 40,))
sol40_T32 = solve_ivp(ODE_system, (0, tend), y0_3, t_eval=t_eval, args=(32, 40,))

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(sol20_T25.t, sol20_T25.y[1],label='T=25, a=25')
ax1.plot(sol40_T25.t, sol40_T25.y[1],label='T=25, a=40')
ax1.plot(sol20_T35.t, sol20_T35.y[1],label='T=35, a=25')
ax1.plot(sol40_T35.t, sol40_T35.y[1],label='T=35, a=40')
ax1.set_xlabel('Time (-)'); ax1.set_ylabel('Leaf Area')
ax1.legend()
ax1.set_ylim(0,50)

ax2.plot(sol20_T25.t, sol20_T25.y[0]/sol20_T25.y[1],label='T=25, a=25')
ax2.plot(sol40_T25.t, sol40_T25.y[0]/sol40_T25.y[1],label='T=25, a=40')
ax2.plot(sol20_T35.t, sol20_T35.y[0]/sol20_T35.y[1],label='T=35, a=25')
ax2.plot(sol40_T35.t, sol40_T35.y[0]/sol40_T35.y[1],label='T=35, a=40')
ax2.set_xlabel('Time (-)'); ax2.set_ylabel('Carbon')
ax2.legend()

fig, ax3 = plt.subplots()
ax3.plot(sol20_T25.t, sol20_T25.y[2], label='T=25, a=20')
ax3.plot(sol40_T25.t, sol40_T25.y[2], label='T=25, a=40')
ax3.plot(sol20_T35.t, sol20_T35.y[2], label='T=35, a=20')
ax3.plot(sol40_T35.t, sol40_T35.y[2], label='T=35, a=40')
ax3.set_xlabel('Time (-)')
ax3.set_ylabel('Leaf Temperature (Tleaf)')
ax3.legend()
ax3.set_title('Leaf Temperature Over Time')
plt.tight_layout()


bar_labels = [
    'angle20_T25', 'angle20_T35',
    'angle40_T25', 'angle40_T35'
]
bar_values = [
    sol20_T25.y[1, -1], sol20_T35.y[1, -1],
    sol40_T25.y[1, -1], sol40_T35.y[1, -1]
]

plt.figure(figsize=(8, 5))
plt.bar(bar_labels, bar_values)
plt.ylabel('Final Leaf Area')
plt.title('Final Leaf Area for Selected Temperatures and Angles')
plt.xticks(rotation=45)
plt.tight_layout()


# Simulate angle 20 and 40 for the temperature range from 20 to 40 C using list comprehension
temps = np.arange(20, 41, 1)
final_LA_angle20 = [solve_ivp(ODE_system, (0, tend), y0_1, t_eval=t_eval, args=(T, 20,)).y[1, -1] for T in temps]
final_LA_angle40 = [solve_ivp(ODE_system, (0, tend), y0_1, t_eval=t_eval, args=(T, 40,)).y[1, -1] for T in temps]

plt.figure(figsize=(8, 5))
plt.plot(temps, final_LA_angle20, label='angle=20', marker='o')
plt.plot(temps, final_LA_angle40, label='angle=40', marker='s')
plt.xlabel('Temperature (C)')
plt.ylabel('Final Leaf Area')
plt.legend()
plt.grid(True)
plt.xlim(20,40)
plt.tight_layout()


plt.show()