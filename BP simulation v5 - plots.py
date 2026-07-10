# MikTeX must be installed on the system!

import numpy as np
import matplotlib.pyplot as plt



""" Parameters """

σ = 0.5                 # substitution elasticitiy 
g = 0.5                 # preference shift 
χ = 3                   # marginal brown cost
ζ = 1                   # additional marginal green cost
m = 0.3                 # fixed cost (m*i)
λ = 8                   # marginal damage of the externality    
β = 0.99                # discount factor
d = 1.5                 # value transition speed
e = 0.25                # technolgy transition speed 
μ0 = 0.25               # initial share of green citizens
γ0 = 0.25               # initial share of green firms
I = 1                   # endowment
gridsize = 50           # grid size for the discretization of μ and γ 
periods = 60


# Deviations for sensitivity analysis
σ_up = 0.51
σ_down = 0.49
g_up = 0.55
g_down = 0.45
χ_up = 3.5
χ_down = 2.5
ζ_up = 1.25
ζ_down = 0.75
m_up = 0.35
m_down = 0.25
λ_up = 10
λ_down = 6
β_up = 1
β_down = 0.97
d_up = 2
d_down = 1
e_up = 0.4
e_down = 0.1
μ0_up = 0.35
μ0_down = 0.15
γ0_up = 0.35
γ0_down = 0.15


parameters_orig = np.array([σ,g,χ,ζ,m,λ,β,d,e,μ0,γ0])       # base case parameters not to be changed
parameters_unicode = ["σ","g","χ","ζ","m","λ","β","d","e","μ0","γ0"]
parameters_tex = ["\sigma","g","\chi","\zeta","m","\lambda",r"\beta","d","e","\mu_0","\gamma_0"]

periods_array = np.arange(periods)      # from 0 to periods-1        
μ_array = np.linspace(0,1,gridsize)     # all μ values 
γ_array = np.linspace(0,1,gridsize)     # all γ values




""" Parameter Initialization for Changed Parameters """

def initialization():                    # sets the parameter variables according to the array "parameters"
    
    global σ,g,χ,ζ,m,λ,β,d,e,μ0,γ0,filename
    
    σ = parameters[0]                 
    g = parameters[1]                 
    χ = parameters[2]                
    ζ = parameters[3]                 
    m = parameters[4]                 
    λ = parameters[5]                 
    β = parameters[6]                 
    d = parameters[7]                 
    e = parameters[8]                 
    μ0 = parameters[9]
    γ0 = parameters[10]
    
    filename = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}' 



""" Plots for the main text """

def plots():
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 16,
    })
    
    # Tax rate
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track[:,2]/Tnorm,  label="$T$, optimized",marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_static[:,2]/Tnorm,  label="$T$, static",marker='o',markevery=5,color='#0072B2')
    plt.xlabel("Years")
    plt.ylabel("Tax rate")
    plt.legend(loc="center left")
    plt.xlim(0,50)
    plt.axis(ymin=0)
    plt.grid()
    plt.savefig(f'plots/{filename} Tax.pdf',bbox_inches='tight')
    
    # Share of green citizens
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track[:,0],label="$\mu$, optimized tax",marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_static[:,0],label="$\mu$, static tax",marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_zero[:,0],label="$\mu$, no tax",marker='X',markevery=5,color='#D55E00')
    plt.xlabel("Years")
    plt.ylabel("Share")
    plt.legend(loc="upper left")
    plt.xlim(0,50)
    plt.ylim((-0.05,1.05))
    plt.grid()
    plt.savefig(f'plots/{filename} Citizens.pdf',bbox_inches='tight')
    
    # Share of green firms
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track[:,1],label="$\gamma$, optimized tax",marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_static[:,1],label="$\gamma$, static tax",marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_zero[:,1],label="$\gamma$, no tax",marker='X',markevery=5,color='#D55E00')
    plt.xlabel("Years")
    plt.ylabel("Share")
    plt.legend(loc="upper left")
    plt.xlim(0,50)
    plt.ylim((-0.05,1.05))
    plt.grid()
    plt.savefig(f'plots/{filename} Firms.pdf',bbox_inches='tight')
    
    # Welfare
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track[:,3]/Wnorm, label="$\Omega$, optimized tax",marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_static[:,3]/Wnorm, label="$\Omega$, static tax",marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_zero[:,3]/Wnorm, label="$\Omega$, no tax",marker='X',markevery=5,color='#D55E00')
    plt.xlabel("Years")
    plt.ylabel("Welfare")
    plt.legend(loc=(0.46,0.17))
    plt.xlim(0,50)
    plt.grid()
    plt.savefig(f'plots/{filename} Welfare.pdf',bbox_inches='tight')

    # Demand for green goods
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track[:,4]/Ynorm, label=r"$\bar{y}$, optimized tax",marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_static[:,4]/Ynorm, label=r"$\bar{y}$, static tax",marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_zero[:,4]/Ynorm, label=r"$\bar{y}$, no tax",marker='X',markevery=5,color='#D55E00')
    plt.xlabel("Years")
    plt.ylabel("Demand")
    plt.legend(loc=(0.47,0.11))
    plt.xlim(0,50)
    plt.grid()
    plt.savefig(f'plots/{filename} GreenDemand.pdf',bbox_inches='tight')

    # Demand for brown goods
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track[:,5]/Ynorm, label=r"$\bar{Y}$, optimized tax",marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_static[:,5]/Ynorm, label=r"$\bar{Y}$, static tax",marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_zero[:,5]/Ynorm, label=r"$\bar{Y}$, no tax",marker='X',markevery=5,color='#D55E00')
    plt.xlabel("Years")
    plt.ylabel("Demand")
    plt.legend(loc=(0.46,0.38))
    plt.xlim(0,50)
    plt.grid()
    plt.savefig(f'plots/{filename} BrownDemand.pdf',bbox_inches='tight')
    


""" Plots for sensitivity analysis """

def sensitivity_plots(i):                   # i is the index of the changed parameter in parameters_unicode
    
    parameter = parameters_unicode[i]                           # e.g. "σ"
    parameter_tex = parameters_tex[i]                           # e.g. "\sigma"
    value_orig = parameters_orig[i]                             # e.g. 0.5 (value of σ)
    value_up = globals()[parameters_unicode[i]+"_up"]           # e.g. 0.51 (value of σ_up)
    value_down = globals()[parameters_unicode[i]+"_down"]       # e.g. 0.49 (value of σ_down)
    
    track_orig = np.load(f'data/{filename_orig} Track.npy')
    track_static_orig = np.load(f'data/{filename_orig} Track_static.npy')
    track_zero_orig = np.load(f'data/{filename_orig} Track_zero.npy')
    
    track_up = np.load(f'data/{filename_up} Track.npy')
    track_static_up = np.load(f'data/{filename_up} Track_static.npy')
    track_zero_up = np.load(f'data/{filename_up} Track_zero.npy')
    
    track_down = np.load(f'data/{filename_down} Track.npy')
    track_static_down = np.load(f'data/{filename_down} Track_static.npy')
    track_zero_down = np.load(f'data/{filename_down} Track_zero.npy')
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 16,
    })
    
    # Tax rate
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track_orig[:,2]/Tnorm, label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,value_orig),marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_up[:,2]/Tnorm, label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,value_up),marker='^',markevery=5,color='#009E73',linestyle="dashed")
    plt.plot(periods_array,track_down[:,2]/Tnorm, label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,value_down),marker='^',markevery=5,color='#009E73',linestyle="dotted")
    plt.plot(periods_array,track_static_orig[:,2]/Tnorm, label=r"$T,~\mathrm{}~{}={}$".format("{static,}",parameter_tex,value_orig),marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_static_up[:,2]/Tnorm, label=r"$T,~\mathrm{}~{}={}$".format("{static,}",parameter_tex,value_up),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
    plt.plot(periods_array,track_static_down[:,2]/Tnorm, label=r"$T,~\mathrm{}~{}={}$".format("{static,}",parameter_tex,value_down),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
    plt.xlabel("Years")
    plt.ylabel("Tax rate")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')#
    plt.xlim(0,50)
    plt.axis(ymin=0)
    plt.grid()
    plt.savefig(f'plots/{filename_orig} {parameter}-sensitivity Tax.pdf',bbox_inches='tight')
    
    # Share of green citizens
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track_orig[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_orig),marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_up[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_up),marker='^',markevery=5,color='#009E73',linestyle="dashed")
    plt.plot(periods_array,track_down[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_down),marker='^',markevery=5,color='#009E73',linestyle="dotted")
    plt.plot(periods_array,track_static_orig[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_orig),marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_static_up[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_up),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
    plt.plot(periods_array,track_static_down[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_down),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
    plt.plot(periods_array,track_zero_orig[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_orig),marker='X',markevery=5,color='#D55E00')
    plt.plot(periods_array,track_zero_up[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_up),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
    plt.plot(periods_array,track_zero_down[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_down),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
    plt.xlabel("Years")
    plt.ylabel("Share")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(0,50)
    plt.ylim((-0.05,1.05))
    plt.grid()
    plt.savefig(f'plots/{filename_orig} {parameter}-sensitivity Citizens.pdf',bbox_inches='tight')
        
    # Share of green firms
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track_orig[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_orig),marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_up[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_up),marker='^',markevery=5,color='#009E73',linestyle="dashed")
    plt.plot(periods_array,track_down[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_down),marker='^',markevery=5,color='#009E73',linestyle="dotted")
    plt.plot(periods_array,track_static_orig[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_orig),marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_static_up[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_up),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
    plt.plot(periods_array,track_static_down[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_down),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
    plt.plot(periods_array,track_zero_orig[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_orig),marker='X',markevery=5,color='#D55E00')
    plt.plot(periods_array,track_zero_up[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_up),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
    plt.plot(periods_array,track_zero_down[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_down),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
    plt.xlabel("Years")
    plt.ylabel("Share")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(0,50)
    plt.ylim((-0.05,1.05))
    plt.grid()
    plt.savefig(f'plots/{filename_orig} {parameter}-sensitivity Firms.pdf',bbox_inches='tight')
    
    # Welfare
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track_orig[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_orig),marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_up[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_up),marker='^',markevery=5,color='#009E73',linestyle="dashed")
    plt.plot(periods_array,track_down[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,value_down),marker='^',markevery=5,color='#009E73',linestyle="dotted")
    plt.plot(periods_array,track_static_orig[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_orig),marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_static_up[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_up),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
    plt.plot(periods_array,track_static_down[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,value_down),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
    plt.plot(periods_array,track_zero_orig[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_orig),marker='X',markevery=5,color='#D55E00')
    plt.plot(periods_array,track_zero_up[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_up),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
    plt.plot(periods_array,track_zero_down[:,3]/Wnorm, label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,value_down),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
    plt.xlabel("Years")
    plt.ylabel("Welfare")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(0,50)
    plt.grid()
    plt.savefig(f'plots/{filename_orig} {parameter}-sensitivity Welfare.pdf',bbox_inches='tight')
    
    # Demand for green goods
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track_orig[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,value_orig),marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_up[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,value_up),marker='^',markevery=5,color='#009E73',linestyle="dashed")
    plt.plot(periods_array,track_down[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,value_down),marker='^',markevery=5,color='#009E73',linestyle="dotted")
    plt.plot(periods_array,track_static_orig[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,value_orig),marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_static_up[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,value_up),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
    plt.plot(periods_array,track_static_down[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,value_down),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
    plt.plot(periods_array,track_zero_orig[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,value_orig),marker='X',markevery=5,color='#D55E00')
    plt.plot(periods_array,track_zero_up[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,value_up),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
    plt.plot(periods_array,track_zero_down[:,4]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,value_down),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
    plt.xlabel("Years")
    plt.ylabel("Demand")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(0,50)
    plt.grid()
    plt.savefig(f'plots/{filename_orig} {parameter}-sensitivity GreenDemand.pdf',bbox_inches='tight')
    
    # Demand for brown goods
    plt.figure(figsize=(6,4))
    plt.plot(periods_array,track_orig[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,value_orig),marker='^',markevery=5,color='#009E73')
    plt.plot(periods_array,track_up[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,value_up),marker='^',markevery=5,color='#009E73',linestyle="dashed")
    plt.plot(periods_array,track_down[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,value_down),marker='^',markevery=5,color='#009E73',linestyle="dotted")
    plt.plot(periods_array,track_static_orig[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,value_orig),marker='o',markevery=5,color='#0072B2')
    plt.plot(periods_array,track_static_up[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,value_up),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
    plt.plot(periods_array,track_static_down[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,value_down),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
    plt.plot(periods_array,track_zero_orig[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,value_orig),marker='X',markevery=5,color='#D55E00')
    plt.plot(periods_array,track_zero_up[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,value_up),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
    plt.plot(periods_array,track_zero_down[:,5]/Ynorm, label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,value_down),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
    plt.xlabel("Years")
    plt.ylabel("Demand")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(0,50)
    plt.grid()
    plt.savefig(f'plots/{filename_orig} {parameter}-sensitivity BrownDemand.pdf',bbox_inches='tight')
    


""" Basecase plots """

parameters = parameters_orig.copy()
initialization()
filename_orig = filename
track = np.load(f'data/{filename} Track.npy')
track_static = np.load(f'data/{filename} Track_static.npy')
track_zero = np.load(f'data/{filename} Track_zero.npy')   

Tnorm = track_static[0,2]
Wnorm = track_static[0,3]
Ynorm = track_static[0,4]

plots()



""" Sensitivity plots """

for i in range(11):                                                             # i is the index of the changed parameter
    parameters[i] = globals()[parameters_unicode[i]+"_up"]                      # before it is parameters = parameters_orig, change of one parameter, e.g. to σ_up
    initialization()                                                            # used to obtain the correct filename to load the tracks.npy for the sensitivity plots
    filename_up = filename
    parameters[i] = globals()[parameters_unicode[i]+"_down"]
    initialization()
    filename_down = filename
    sensitivity_plots(i)
    parameters = parameters_orig.copy()

