# Code created by Monica Garcia Gomez for Modeling Life course, Utrecht University
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap
import os
os.makedirs("output", exist_ok=True)

def ODErun(ODEtested,nodeA,nodeB):
    times = np.arange(0, 5000.1, 0.1)
    IC=0
    parameters = {'prod': 0.01, 'decay': 0.001,'Ksat': 5, 'n': 2,'nodeA':nodeA,'nodeB':nodeB} # here you control the value of the inputs: nodeA and nodeB
    cells = odeint(ODEtested, IC, times, args=(parameters,)) #np.shape
    plt.figure(figsize=(8, 4))
    plt.plot(times, cells, label='outputC concentration')
    plt.xlabel('Time')
    plt.ylabel('outputC')
    plt.title('outputC dynamics (ODE model)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #WC1 AND OR

def ODEBooleanPlot(ode_output, bool_output):
    # Plot results side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ODE output
    sns.heatmap(np.flipud(ode_output), ax=axes[0], xticklabels=20, yticklabels=20, cmap="viridis")
    axes[0].set_xlabel('nodeB')
    axes[0].set_ylabel('nodeA')
    axes[0].set_title('outputC steady-state concentration')
    axes[0].set_yticks(np.linspace(0, 10, 6))
    axes[0].set_yticklabels([str(int(x)) for x in np.linspace(10, 0, 6)])

    # Boolean output
    sns.heatmap(np.flipud(bool_output), ax=axes[1], xticklabels=[0, 1], yticklabels=[1, 0], cmap="viridis")
    axes[1].set_xlabel('nodeB (Boolean)')
    axes[1].set_ylabel('nodeA (Boolean)')
    axes[1].set_title('Boolean gate output')
    axes[1].set_yticks(np.linspace(0, 1, 6))
    axes[1].set_yticklabels([str(int(x)) for x in np.linspace(1, 0, 6)])
    plt.savefig("output/0-circuits-logical-gates.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # If you want a fun way of plotting ODE results, use the following commented code: 
    # ODE plot 3D, you can move it around. 
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(11), np.arange(11))
    ax3d.plot_surface(X, Y, ode_output, cmap='viridis', edgecolor='none')
    ax3d.set_xlabel('nodeB')
    ax3d.set_ylabel('nodeA')
    ax3d.set_zlabel('outputC')
    ax3d.set_title('3D surface: outputC steady-state')
    plt.tight_layout()
    plt.show()

def rootNetwork(parameters): 
    CK=parameters['CK']
    ARR1=parameters['ARR1']
    SHY2=parameters['SHY2']
    AUXIAAR=parameters['AUXIAAR']
    ARFR=parameters['ARFR']
    ARF10=parameters['ARF10']
    ARF5=parameters['ARF5']
    XAL1=parameters['XAL1']
    PLT=parameters['PLT']
    AUX=parameters['AUX']
    SCR=parameters['SCR']
    SHR=parameters['SHR']
    MIR165=parameters['MIR165']
    PHB=parameters['PHB']
    JKD=parameters['JKD']
    MGP=parameters['MGP']
    WOX5=parameters['WOX5']
    CLE40=parameters['CLE40']
    w_CK = (PHB and (1-ARFR) or (1-SHR))
    w_ARR1 = CK and (1-SCR)
    w_SHY2 = (1-AUX) and ARR1
    w_AUXIAAR = 1-AUX
    w_ARFR = 1-AUXIAAR
    w_ARF10 = (1-AUXIAAR) and (1-(JKD and SHR))
    w_ARF5 = (PHB or PLT) and (1-(SHR and MGP)) and (1-AUXIAAR)and (1-SHY2)
    w_XAL1= ARFR
    w_PLT= ARF5 or XAL1 or ARFR or WOX5
    w_AUX = 1
    w_SCR = SHR and SCR and JKD
    w_SHR = SHR
    w_MIR165 = (1-PHB) or (SCR and SHR and (1-ARR1))
    w_PHB = (((1-ARR1) and PLT) or PHB) and  (1-MIR165)
    w_JKD = (1-PHB) and SCR and SHR
    w_MGP = (1-ARF5) and SCR and SHR
    w_WOX5 = (1-ARF10) and ARF5 and (1-CLE40)
    w_CLE40 = 1-SHR
    return(w_CK,w_ARR1,w_SHY2,w_AUXIAAR,w_ARFR,w_ARF10,w_ARF5,w_XAL1,w_PLT,w_AUX,w_SCR,w_SHR,w_MIR165,w_PHB,w_JKD,w_MGP,w_WOX5,w_CLE40)

def rootNetworkAsynchronous(parameters):
    CK=parameters['CK']
    ARR1=parameters['ARR1']
    SHY2=parameters['SHY2']
    AUXIAAR=parameters['AUXIAAR']
    ARFR=parameters['ARFR']
    ARF10=parameters['ARF10']
    ARF5=parameters['ARF5']
    XAL1=parameters['XAL1']
    PLT=parameters['PLT']
    AUX=parameters['AUX']
    SCR=parameters['SCR']
    SHR=parameters['SHR']
    MIR165=parameters['MIR165']
    PHB=parameters['PHB']
    JKD=parameters['JKD']
    MGP=parameters['MGP']
    WOX5=parameters['WOX5']
    CLE40=parameters['CLE40']
    # With 0.5 probability, set w_CK as before, otherwise set to 0
    if np.random.rand() < 0.95:
        w_CK = (PHB and (1-ARFR) or (1-SHR))
    else:
        w_CK = CK

    if np.random.rand() < 0.95:
        w_ARR1 = CK and (1-SCR)
    else:
        w_ARR1 = ARR1

    if np.random.rand() < 0.95:
        w_SHY2 = (1-AUX) and ARR1
    else:
        w_SHY2 = SHY2

    if np.random.rand() < 0.95:
        w_AUXIAAR = 1-AUX
    else:
        w_AUXIAAR = AUXIAAR

    if np.random.rand() < 0.95:
        w_ARFR = 1-AUXIAAR
    else:
        w_ARFR = ARFR

    if np.random.rand() < 0.95:
        w_ARF10 = (1-AUXIAAR) and (1-(JKD and SHR))
    else:
        w_ARF10 = ARF10

    if np.random.rand() < 0.95:
        w_ARF5 = (PHB or PLT) and (1-(SHR and MGP)) and (1-AUXIAAR) and (1-SHY2)
    else:
        w_ARF5 = ARF5

    if np.random.rand() < 0.95:
        w_XAL1 = ARFR
    else:
        w_XAL1 = XAL1

    if np.random.rand() < 0.95:
        w_PLT = ARF5 or XAL1 or ARFR or WOX5
    else:
        w_PLT = PLT

    if np.random.rand() < 0.95:
        w_AUX = 1
    else:
        w_AUX = AUX

    if np.random.rand() < 0.95:
        w_SCR = SHR and SCR and JKD
    else:
        w_SCR = SCR

    if np.random.rand() < 0.95:
        w_SHR = SHR
    else:
        w_SHR = SHR

    if np.random.rand() < 0.95:
        w_MIR165 = (1-PHB) or (SCR and SHR and (1-ARR1))
    else:
        w_MIR165 = MIR165

    if np.random.rand() < 0.95:
        w_PHB = (((1-ARR1) and PLT) or PHB) and (1-MIR165)
    else:
        w_PHB = PHB

    if np.random.rand() < 0.95:
        w_JKD = (1-PHB) and SCR and SHR
    else:
        w_JKD = JKD

    if np.random.rand() < 0.95:
        w_MGP = (1-ARF5) and SCR and SHR
    else:
        w_MGP = MGP

    if np.random.rand() < 0.95:
        w_WOX5 = (1-ARF10) and ARF5 and (1-CLE40)
    else:
        w_WOX5 = WOX5

    if np.random.rand() < 0.95:
        w_CLE40 = 1-SHR
    else:
        w_CLE40 = CLE40
    return(w_CK,w_ARR1,w_SHY2,w_AUXIAAR,w_ARFR,w_ARF10,w_ARF5,w_XAL1,w_PLT,w_AUX,w_SCR,w_SHR,w_MIR165,w_PHB,w_JKD,w_MGP,w_WOX5,w_CLE40)

def plotBooleanTimecourse(matrix,timesteps):
    plt.figure(figsize=(16, 8))
    node_names = ['CK', 'ARR1', 'SHY2', 'AUXIAAR', 'ARFR', 'ARF10', 'ARF5', 'XAL1', 
                  'PLT', 'AUX', 'SCR', 'SHR', 'MIR165', 'PHB', 'JKD', 'MGP', 'WOX5', 'CLE40']
    sns.heatmap(matrix, cmap="viridis", cbar=True, xticklabels=node_names, cbar_kws={"ticks": [0, 1]})
    #sns.heatmap(matrix, cmap=sns.color_palette(["red", "green"], as_cmap=True), cbar=True, xticklabels=node_names, cbar_kws={"ticks": [0, 1]})
    plt.grid(visible=True, which='both', axis='both', color='white', linewidth=0.5)
    plt.xticks(np.arange(len(node_names)), node_names, rotation=90)
    plt.yticks(np.arange(timesteps+1), np.arange(timesteps+1))
    plt.xlabel("Nodes")
    plt.ylabel("Timesteps")
    plt.title("Boolean Network Timecourse")
    plt.savefig("output/1-boolean-IC-timecourse.png", dpi=300, bbox_inches='tight')
    plt.show()

def plotBooleanAttractors(attractors):
    # here we plot the attractors found
    plt.figure(figsize=(16, 8))
    node_names = ['CK', 'ARR1', 'SHY2', 'AUXIAAR', 'ARFR', 'ARF10', 'ARF5', 'XAL1', 
                  'PLT', 'AUX', 'SCR', 'SHR', 'MIR165', 'PHB', 'JKD', 'MGP', 'WOX5', 'CLE40']
    sns.heatmap(attractors, cmap="viridis", cbar=True, xticklabels=node_names, cbar_kws={"ticks": [0, 1]})
    #sns.heatmap(attractors, cmap=sns.color_palette(["red", "green"], as_cmap=True), cbar=True, xticklabels=node_names,cbar_kws={"ticks": [0, 1]})
    plt.grid(visible=True, which='both', axis='both', color='white', linewidth=0.5)
    plt.xticks(np.arange(len(node_names)), node_names, rotation=90)
    ICs = attractors.shape[0]
    plt.yticks(np.arange(ICs), )
    plt.xlabel("Nodes")
    plt.ylabel("Initial Conditions")
    plt.title("Recovered Attractors")
    plt.savefig("output/1-boolean-many-ICs.png", dpi=300, bbox_inches='tight')
    plt.show()

def UMAPBoolean(attractors):
    # 3.1 UMAP dimensionality reduction - for this it is better to increase the number of ICs to 500
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='hamming', random_state=42)
    attractors_umap = reducer.fit_transform(attractors)

    plt.figure(figsize=(11, 8))
    # Color cells by their attractor state (e.g., by the sum of active nodes)
    colors = np.sum(attractors, axis=1)
    # If the second to last position in attractor is 1, color that point purple
    special_mask = attractors[:, -2] == 1
    scatter = plt.scatter(attractors_umap[~special_mask, 0], attractors_umap[~special_mask, 1], edgecolor='k')
    plt.scatter(
        attractors_umap[(attractors[:, 11] == 1) & (attractors[:, 13] == 1), 0],
        attractors_umap[(attractors[:, 11] == 1) & (attractors[:, 13] == 1), 1],
        color='red', edgecolor='k', label='Vascular 1'
    )
    plt.scatter(
        attractors_umap[(attractors[:, 12] == 1) & (attractors[:, 11] == 1), 0],
        attractors_umap[(attractors[:, 12] == 1) & (attractors[:, 11] == 1), 1],
        color='yellow', edgecolor='k', label='Vascular 2'
    )
    plt.scatter(
        attractors_umap[(attractors[:, 17] == 1) & (attractors[:, 13] == 1), 0],
        attractors_umap[(attractors[:, 17] == 1) & (attractors[:, 13] == 1), 1],
        color='orange', edgecolor='k', label='Vascular 3'
    )
    plt.scatter(
        attractors_umap[(attractors[:, 17] == 1) & (attractors[:, 12] == 1), 0],
        attractors_umap[(attractors[:, 17] == 1) & (attractors[:, 12] == 1), 1],
        color='brown', edgecolor='k', label='Columella'
    )
    plt.scatter(
        attractors_umap[(attractors[:, 15] == 1) & (attractors[:, 14] == 1), 0],
        attractors_umap[(attractors[:, 15] == 1) & (attractors[:, 14] == 1), 1],
        color='green', edgecolor='k', label='Endodermis'
    )
    plt.scatter(attractors_umap[special_mask, 0], attractors_umap[special_mask, 1], color='violet', edgecolor='k', label='QC')
    plt.legend()
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP projection of attractors")
    plt.grid(True)
    plt.savefig("output/1-boolean-UMAP.png", dpi=300, bbox_inches='tight')
    plt.show()

def rootNetworkODE(a,t,parameters): 
    decayrate=parameters['decayrate'] # decay rate
    h=parameters['h'] # this controls the shape of the sigmoid function
    CK=a[0]
    ARR1=a[1]
    SHY2=a[2]
    AUXIAAR=a[3]
    ARFR=a[4]
    ARF10=a[5]
    ARF5=a[6]
    XAL1=a[7]
    PLT=a[8]
    AUX=a[9]
    SCR=a[10]
    SHR=a[11]
    MIR165=a[12]
    PHB=a[13]
    JKD=a[14]
    MGP=a[15]
    WOX5=a[16]
    CLE40=a[17]
    w_CK = max(min(PHB,1-ARFR),1-SHR)
    w_ARR1 = min(CK,1-SCR)
    w_SHY2 = min(1-AUX,ARR1)
    w_AUXIAAR = 1-AUX
    w_ARFR = 1-AUXIAAR
    w_ARF10 = min(1-AUXIAAR,1-min(JKD,SHR))
    w_ARF5 = min(max(PHB,PLT),1-min(SHR,MGP),1-AUXIAAR,1-SHY2)
    w_XAL1= ARFR
    w_PLT= max(ARF5,XAL1,ARFR,WOX5)
    w_AUX = 1
    w_SCR = min(SHR,SCR,JKD)
    w_SHR = SHR
    w_MIR165 = max(1-PHB,min(SCR,SHR,1-ARR1))
    w_PHB = min(max(min(1-ARR1,PLT),PHB),1-MIR165)
    w_JKD = min(1-PHB,SCR,SHR)
    w_MGP = min(1-ARF5,SCR,SHR)
    w_WOX5 = min(1-ARF10,ARF5,1-CLE40)
    w_CLE40 = 1-SHR
    def sigmoid(w_node,h):
        a=((-np.exp(0.5*h)+np.exp(-h*(w_node)))/((1-np.exp(0.5*h))*(1+np.exp(-h*(w_node-0.5)))))
        return(a)
    dCK = sigmoid(w_CK,h)-(decayrate*CK)
    dARR1 = sigmoid(w_ARR1,h)-(decayrate*ARR1)
    dSHY2 = sigmoid(w_SHY2,h)-(decayrate*SHY2)
    dAUXIAAR = sigmoid(w_AUXIAAR,h)-(decayrate*AUXIAAR)
    dARFR = sigmoid(w_ARFR,h)-(decayrate*ARFR)
    dARF10 = sigmoid(w_ARF10,h)-(decayrate*ARF10)
    dARF5 = sigmoid(w_ARF5,h)-(decayrate*ARF5)
    dXAL1 = sigmoid(w_XAL1,h)-(decayrate*XAL1)
    dPLT = sigmoid(w_PLT,h)-(decayrate*PLT)
    dAUX = sigmoid(w_AUX,h)-(decayrate*AUX)
    dSCR = sigmoid(w_SCR,h)-(decayrate*SCR)
    dSHR = sigmoid(w_SHR,h)-(decayrate*SHR)
    dMIR165 = sigmoid(w_MIR165,h)-(decayrate*MIR165)
    dPHB = sigmoid(w_PHB,h)-(decayrate*PHB)
    dJKD = sigmoid(w_JKD,h)-(decayrate*JKD)
    dMGP = sigmoid(w_MGP,h)-(decayrate*MGP)
    dWOX5 = sigmoid(w_WOX5,h)-(decayrate*WOX5)
    dCLE40 = sigmoid(w_CLE40,h)-(decayrate*CLE40)	       
    return(dCK, dARR1, dSHY2, dAUXIAAR, dARFR, dARF10, dARF5, dXAL1, dPLT, dAUX, dSCR, dSHR, dMIR165, dPHB, dJKD, dMGP, dWOX5, dCLE40)

def plotODEroot(cells,times):
    attractorode = cells[-1, :]  # last value of the ODE simulation

    nodesNames=['CK','ARR1','SHY2','AUXIAAR','ARFR','ARF10','ARF5','XAL1','PLT','AUX','SCR','SHR','MIR165','PHB','JKD','MGP','WOX5','CLE40']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Time series plot
    for i, name in enumerate(nodesNames):
        ax1.plot(times, cells[:, i], label=name, linewidth=3, color=sns.color_palette("tab20", len(nodesNames))[i])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Gene activity')
    ax1.set_title('Root model (ODE version)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    ax1.grid(True)

    # Heatmap of final state
    sns.heatmap(attractorode.reshape(-1, 1), yticklabels=nodesNames, cmap="viridis", ax=ax2, cbar=False)
    #sns.heatmap(attractorode.reshape(-1, 1), yticklabels=nodesNames, cmap=sns.color_palette(["red", "green"], as_cmap=True), ax=ax2, cbar=False)
    ax2.set_xticks([])
    ax2.set_ylabel('Nodes')
    ax2.set_title('Final state')
    ax2.set_aspect(0.7)  # Make the heatmap narrower
    plt.savefig("output/2-boolean-to-ode-dynamics.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plotODErootTransition(cells,times):
    attractorode = cells[-1, :]  # last value of the ODE simulation
    nodesNames=['CK','ARR1','SHY2','AUXIAAR','ARFR','ARF10','ARF5','XAL1','PLT','AUX','SCR','SHR','MIR165','PHB','JKD','MGP','WOX5','CLE40']
    end=[1,1,0,0,1,1,1,1,1,1,0,0,1,0,0,0,0,1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1, 1]})

    # Time series plot
    for i, name in enumerate(nodesNames):
        ax1.plot(times, cells[:, i], label=name, linewidth=3, color=sns.color_palette("tab20", len(nodesNames))[i])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Gene activity')
    ax1.set_title('Root model (ODE version)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    ax1.grid(True)

    # Heatmap of final state
    sns.heatmap(attractorode.reshape(-1, 1), yticklabels=nodesNames, cmap="viridis", ax=ax2, cbar=False)
    #sns.heatmap(attractorode.reshape(-1, 1), yticklabels=nodesNames, cmap=sns.color_palette(["red", "green"], as_cmap=True), ax=ax2, cbar=False)
    ax2.set_xticks([])
    ax2.set_ylabel('Nodes')
    ax2.set_title('Final state')
    ax2.set_aspect(0.7)  # Make the heatmap narrower

    # Heatmap of desired end state
    sns.heatmap(np.array(end).reshape(-1, 1), yticklabels=nodesNames, cmap="viridis", ax=ax3, cbar=False)
    #sns.heatmap(np.array(end).reshape(-1, 1), yticklabels=nodesNames, cmap=sns.color_palette(["red", "green"], as_cmap=True), ax=ax3, cbar=False)
    ax3.set_xticks([])
    ax3.set_ylabel('Nodes')
    ax3.set_title('Desired end state')
    ax3.set_aspect(0.7)
    plt.savefig("output/2-boolean-to-ode-cell-fate-changes.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
