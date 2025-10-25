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

def Photosynthesis(LA, T,angle,light,t):

    # Photosynthesis rate depends on leaf Temperature
    TempPhoto = Photosynthesis_per_m2(T)

    #Photosynthesis goes down with increased leaf angle - assuming light comes from the top
    LA_eff = EffectiveLeafArea(LA, angle)
    
    #Max Photosynthesis rate
    Pmax = 1

    return Pmax * TempPhoto *  LA_eff * light
    
def Respiration(C_conc,LA,T,Q10):
    #Respiration rate at 25C
    R25 = 0.2 

    TempResp    = R25 * Respiration_per_m2(T, Q10)
    return TempResp *  LA * (1-0.5)*C_conc #0.125 does better   

def Growth(C_conc, LA, t):
    #LA increase saturates with Carbon concentration in the leaf
    growthrate= growth_daynight(t)
    growth=growthrate*(0.25 * C_conc/(C_conc+2)) * LA
    return growth

def T_daynight(T, t):
    #return (max(0, 2*T / 3 + T /3  * np.sin(2 * np.pi * (t-6) / 24))) #1/3 to 1
    #return (max(0, 3*T / 4 + T / 4 * np.sin(2 * np.pi * (t-6) / 24))) # 2/4 to 1
    #return (max(0, 4*T / 5 + T /5  * np.sin(2 * np.pi * (t-6) / 24))) # 3/5 to 1
    #return (max(0, 7*T / 8 + T /8  * np.sin(2 * np.pi * (t-6) / 24))) # 6/8 to 1
    return (max(0, 8*T / 9 + T /9  * np.sin(2 * np.pi * (t-6) / 24))) # 6/8 to 1

def angle_daynight(angle, t):
    #return max(0, 3*angle/4 + angle/4 * np.sin(2 * np.pi * (t -14) / 24))
    return max(0, 2*angle/3 + angle/3 * np.sin(2 * np.pi * (t -14) / 24))
    
def light_daynight(t):
    return max(0, 2 * np.sin(2 * np.pi * (t-6) / 24))

def growth_daynight(t):
    base=0.6#0.7
    A=0.4#0.3
    growthrate= base + A * np.sin(2 * np.pi * (t -18) / 24) #-18
    return growthrate
   
# Time loop over 5 days
days = 5
time_steps = np.linspace(0, 24 * days, 1000)  # 5 days with 1000 time steps
temperature = [T_daynight(35, t % 24) for t in time_steps]  # Example base temperature of 30°C
angles = [angle_daynight(40, t % 24) for t in time_steps]  # Example base angle of 30°
light_levels = [light_daynight(t % 24) for t in time_steps]  # Light levels over the day
growth_rates = [growth_daynight(t % 24) for t in time_steps]  # Growth rates over time

# Plot temperature, angle, light levels, and growth rates over time in a single figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Add grey shading for night times (18:00 to 6:00)
for day in range(days):
    ax1.axvspan(day * 24 + 18, day * 24 + 24, color='grey', alpha=0.3, label='Night' if day == 0 else None)
    ax1.axvspan(day * 24, day * 24 + 6, color='grey', alpha=0.3)

# Plot temperature and angles on the primary y-axis
ax1.plot(time_steps, temperature, label='Temperature (°C)', color='red')
ax1.plot(time_steps, angles, label='Leaf Angle (°)', color='green')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Temperature (°C) / Leaf Angle (°)')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a secondary y-axis for light levels and growth rates
ax2 = ax1.twinx()
ax2.plot(time_steps, light_levels, label='Light Levels', color='blue')
ax2.plot(time_steps, growth_rates, label='Growth Rate', color='purple', linestyle='--')
ax2.set_ylabel('Light Levels / Growth Rate')
ax2.legend(loc='upper right')

plt.title('Temperature, Leaf Angle, Light Levels, and Growth Rate Over Time')
plt.tight_layout()
plt.show()



def ODE_system(t, y, T, angle):
    C, LA = y;    C_conc=C/LA

    temp = T_daynight(T,t)
    an = angle_daynight(angle, t)
    light = light_daynight(t)
    T_leaf = temp*(1 - 0.3*(1-np.cos(np.radians(an))))
    dCdt =  Photosynthesis(LA, temp,an,light,T_leaf) - Respiration(C_conc,LA,T_leaf,2.5) - Growth(C_conc,LA,t)
    #dCdt =  Photosynthesis(LA, temp,an,light,t) - Respiration(LA,temp,2.5) - Growth(C_conc,LA)
    dLAdt = Growth(C_conc,LA,t)

    return [dCdt, dLAdt]

""""Thee thermomorphogenesis repsonse does also have a positive effect. It is shown to lower leaf temperature, 
which increases photosynthesis and decreases respiration.Investigate if this can have an effect, 
i.e. how big is the effect for 3/5/7 degrees cooling down?"""

#Investigate the effect of leaf hyponasty
y0 = [2.5, 2.5]  #Initial LA and C state of the plant
# Time span of the simulation
tend=5*24; t_eval = np.linspace(0, tend, 5*1000)

#Two temperatures to test
T2=25;T3=35
#Two angles to test
angle1 = 20; angle2 = 40

#plot final LA in a barplot
#1 to 3 degree cooling bar plot
# Collect final leaf area values for specified temperatures and angles
sol20_T25 = solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(25, 20,))
sol20_T35 = solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(35, 20,))
sol40_T25 = solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(25, 40,))
sol40_T35 = solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(35, 40,))
sol40_T34 = solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(34, 40,))
sol40_T33 = solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(33, 40,))
sol40_T32 = solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(32, 40,))

#Fig1
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


plt.figure()
plt.plot(sol20_T25.t,[T_daynight(25,t) for t in sol20_T25.t] )
plt.plot(sol20_T25.t,[T_daynight(35,t) for t in sol20_T25.t] )

plt.ylabel('T')


bar_labels = [
    'angle20_T25', 'angle20_T35',
    'angle40_T25', 'angle40_T35',
    'angle40_T34', 'angle40_T33', 'angle40_T32'
]
bar_values = [
    sol20_T25.y[1, -1], sol20_T35.y[1, -1],
    sol40_T25.y[1, -1], sol40_T35.y[1, -1],
    sol40_T34.y[1, -1], sol40_T33.y[1, -1], sol40_T32.y[1, -1]
]

plt.figure(figsize=(8, 5))
plt.bar(bar_labels, bar_values)
plt.ylabel('Final Leaf Area')
plt.title('Final Leaf Area for Selected Temperatures and Angles')
plt.xticks(rotation=45)
plt.tight_layout()


# Simulate angle 20 and 40 for the temperature range from 20 to 40 C using list comprehension
#Fig 4
temps = np.arange(20, 41, 1)
final_LA_angle20 = [solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(T, 20,)).y[1, -1] for T in temps]
final_LA_angle40 = [solve_ivp(ODE_system, (0, tend), y0, t_eval=t_eval, args=(T, 40,)).y[1, -1] for T in temps]

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
