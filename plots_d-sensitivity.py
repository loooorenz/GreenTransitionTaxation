# MikTeX must be installed on the system!
import numpy as np
import matplotlib.pyplot as plt


""" Parameters """

σ = 0.5                 # substitution elasticitiy 
g = 0.5                 # preference shift 
χ = 3.0                 # marginal brown cost
ζ = 1.0                 # additional marginal green cost
m = 0.3                 # fixed cost (m*i)
λ = 8.0                 # marginal damage of the externality    
β = 0.99                # discount factor
d = 1.5                 # value transition speed
e = 0.25                # technolgy transition speed 
μ0 = 0.25               # initial share of green citizens
γ0 = 0.25               # initial share of green firms
I = 1                   # endowment
gridsize = 50           # grid size for the discretization of μ and γ 
periods = 60

periods_array = np.arange(periods)      # from 0 to periods-1 

filename = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}'
track_static = np.load(f'data/{filename} Track_static.npy')

Tnorm = track_static[0,2]
Wnorm = track_static[0,3]
Ynorm = track_static[0,4]



""" Importing tracks for different values of d """

d = 0.0
filename00 = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}'
d = 0.5
filename05 = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}'
d = 1.0
filename10 = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}'
d = 1.5
filename15 = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}'
d = 2.0
filename20 = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}'

parameter = "d"                           
parameter_tex = "d" 

track00 = np.load(f'data/{filename00} Track.npy')
track_static00 = np.load(f'data/{filename00} Track_static.npy')
track_zero00 = np.load(f'data/{filename00} Track_zero.npy')
track05 = np.load(f'data/{filename05} Track.npy')
track_static05 = np.load(f'data/{filename05} Track_static.npy')
track_zero05 = np.load(f'data/{filename05} Track_zero.npy')
track10 = np.load(f'data/{filename10} Track.npy')
track_static10 = np.load(f'data/{filename10} Track_static.npy')
track_zero10 = np.load(f'data/{filename10} Track_zero.npy')        
track15 = np.load(f'data/{filename15} Track.npy')
track_static15 = np.load(f'data/{filename15} Track_static.npy')
track_zero15 = np.load(f'data/{filename15} Track_zero.npy')       
track20 = np.load(f'data/{filename20} Track.npy')
track_static20 = np.load(f'data/{filename20} Track_static.npy')
track_zero20 = np.load(f'data/{filename20} Track_zero.npy')     




""" Model functions """

t = -σ*(χ+ζ)                            # green tax is fixed to the static green tax
T_static = (1-σ)*λ-σ*χ                  # static brown tax

def Ω(T,μ,γ):                           # negative welfare function (to use minimization instead of maximization)            
    return -(γ*(1+μ*g)*w() + (1-γ)*(1-μ*g)*W(T) + I - γ**2 * m/2)

def w():                                # part of the welfare function 
    return κ(ζ+t)/(1-σ) - (χ+ζ) * κ(ζ+t)**(1/(1-σ))

def W(T):                               # part of the welfare function 
    return κ(T)/(1-σ) -  (χ+λ) * κ(T)**(1/(1-σ))

def μ_(T,μ,γ):                          # transition function for share of green citizens 
    return min(max(μ + μ*(1-μ)*d*Δ(T,γ), 0), 1)

def Δ(T,γ):                             # green utility advantage
    return σ*g/(1-σ) * (γ * κ(ζ+t) - (1-γ) * κ(T))

def γ_(T,μ,γ):                          # transition function for share of green firms 
    return min(max(γ + e * ((σ/m)*((1+μ*g)*κ(ζ+t) - (1-μ*g)*κ(T)) - γ), 0), 1) 
    
def κ(x):                       
    return ((χ+x)/(1-σ))**(1-(1/σ))

def y(μ,γ):                             # aggregated demand for green goods \bar{y}
    return γ * (1+μ*g) * ((χ+ζ+t)/(1-σ))**(-1/σ)

def Y(T,μ,γ):                           # aggregated demand for brown goods \bar{Y}
    return (1-γ) * (1-μ*g) * ((χ+T)/(1-σ))**(-1/σ)



""" Tracking the development for d=1.5 when the optimized tax for d=0 (ignoring changing preferences) is imposed """

d = 1.5

track_ignore = np.zeros((periods,6))       # axis 1: [0]: μ, [1]: γ, [2]: T, [3]: Ω, [4]: y, [5]: Y
period = 0
μ = μ0
γ = γ0

while period < periods:
    track_ignore[period,0] = μ
    track_ignore[period,1] = γ
    track_ignore[period,2] = track00[period,2]
    track_ignore[period,3] = -Ω(track_ignore[period,2],μ,γ)
    track_ignore[period,4] = y(μ,γ)
    track_ignore[period,5] = Y(track_ignore[period,2],μ,γ)
    μ = μ_(track_ignore[period,2],μ,γ)
    γ = γ_(track_ignore[period,2],μ,γ)
    period += 1



""" Plots for the main text """

plt.rcParams.update({
"text.usetex": True,
"font.family": "Times New Roman",
"font.size": 16,
})

# Tax rate
plt.figure(figsize=(6,4))
plt.plot(periods_array,track15[:,2]/Tnorm, label="$T$, optimized",marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track00[:,2]/Tnorm, label="$T$, optimized for $d=0$",marker='^',markevery=5,color='#009E73',linestyle="dashed")
#plt.plot(periods_array,track_static10[:,2]/Tnorm,label="$T$ static",marker='o',markevery=5,color='#0072B2')
plt.xlabel("Years")
plt.ylabel("Tax rate")
plt.legend(loc=(0.36,0.01))
plt.xlim(0,50)
plt.axis(ymin=0)
plt.grid()
plt.savefig(f'plots/{filename15} extra Tax.pdf',bbox_inches='tight')       

# Welfare
plt.figure(figsize=(6,4))
plt.plot(periods_array,track15[:,3]/Wnorm,label="$\Omega$, optimized tax",marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track_ignore[:,3]/Wnorm,label="$\Omega$, optimized tax for $d=0$",marker='^',markevery=5,color='#009E73',linestyle="dashed")   
#plt.plot(periods_array,track_static10[:,3]/Wnorm,label="$\Omega$, static tax",marker='o',markevery=5,color='#0072B2')
plt.xlabel("Years")
plt.ylabel("Welfare")
plt.legend(loc=(0.28,0.01))
plt.xlim(0,50)
plt.grid()
plt.savefig(f'plots/{filename15} extra Welfare.pdf',bbox_inches='tight')



""" Plots for sensitivity analysis """

# Tax rate 
plt.figure(figsize=(6,4))
plt.plot(periods_array,track15[:,2]/Tnorm,label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,1.5),marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track20[:,2]/Tnorm,label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,2.0),marker='^',markevery=5,color='#009E73',linestyle="dashed")
plt.plot(periods_array,track10[:,2]/Tnorm,label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,1.0),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
plt.plot(periods_array,track05[:,2]/Tnorm,label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,0.5),marker='^',markevery=5,color='#009E73',linestyle=(0,(5,5)))
plt.plot(periods_array,track00[:,2]/Tnorm,label=r"$T,~\mathrm{}~{}={}$".format("{optimized,}",parameter_tex,0.0),marker='^',markevery=5,color='#009E73',linestyle="dotted")
plt.plot(periods_array,track_static10[:,2]/Tnorm,label=r"$T,~\mathrm{}~\forall d$".format("{static,}"),marker='o',markevery=5,color='#0072B2')
plt.xlabel("Years")
plt.ylabel("Tax rate")
plt.axis(ymin=0)
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.xlim(0,50)
plt.grid()
plt.savefig(f'plots/{filename15} {parameter}-sensitivity extra Tax.pdf',bbox_inches='tight')         

# Share of green citizens
plt.figure(figsize=(6,4))
plt.plot(np.linspace(0,50,11),np.full((11,1),0.25),label=r"$\mu,~d=0.0$",color="k",marker='.')
plt.plot(periods_array,track15[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,1.5),marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track20[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,2.0),marker='^',markevery=5,color='#009E73',linestyle="dashed")
plt.plot(periods_array,track10[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,1.0),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
#plt.plot(periods_array,track05[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,0.5),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
#plt.plot(periods_array,track00[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,0.0),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
plt.plot(periods_array,track_static15[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2')
plt.plot(periods_array,track_static20[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,2.0),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
plt.plot(periods_array,track_static10[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,1.0),marker='o',markevery=5,color='#0072B2',linestyle=(0,(1,5)))
#plt.plot(periods_array,track_static05[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,0.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(1,6)))
#plt.plot(periods_array,track_static00[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,0.0),marker='o',markevery=5,color='#0072B2',linestyle=(0,(1,6)))
plt.plot(periods_array,track_zero15[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00')
plt.plot(periods_array,track_zero20[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,2.0),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
plt.plot(periods_array,track_zero10[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,1.0),marker='X',markevery=5,color='#D55E00',linestyle=(0,(1,5)))
#plt.plot(periods_array,track_zero05[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,0.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(1,7)))
#plt.plot(periods_array,track_zero00[:,0],label=r"$\mu,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,0.0),marker='X',markevery=5,color='#D55E00',linestyle=(0,(1,7)))
plt.xlabel("Years")
plt.ylabel("Share")
plt.ylim((-0.05,1.05))
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.xlim(0,50)
plt.grid()
plt.savefig(f'plots/{filename15} {parameter}-sensitivity extra Citizens.pdf',bbox_inches='tight')

# Share of green firms
plt.figure(figsize=(6,4))
plt.plot(periods_array,track15[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,1.5),marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track20[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,2.0),marker='^',markevery=5,color='#009E73',linestyle="dashed")
plt.plot(periods_array,track10[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,1.0),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
#plt.plot(periods_array,track05[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,0.5),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
plt.plot(periods_array,track00[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,0.0),marker='^',markevery=5,color='#009E73',linestyle="dotted")
plt.plot(periods_array,track_static15[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2')
plt.plot(periods_array,track_static20[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,2.0),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
#plt.plot(periods_array,track_static10[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,1.0),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
#plt.plot(periods_array,track_static05[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,0.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(1,6)))
plt.plot(periods_array,track_static00[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,0.0),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
plt.plot(periods_array,track_zero15[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00')
plt.plot(periods_array,track_zero20[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,2.0),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
#plt.plot(periods_array,track_zero10[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,1.0),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
#plt.plot(periods_array,track_zero05[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,0.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(1,7)))
plt.plot(periods_array,track_zero00[:,1],label=r"$\gamma,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,0.0),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
plt.xlabel("Years")
plt.ylabel("Share")
plt.ylim((-0.05,1.05))
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.xlim(0,50)
plt.grid()
plt.savefig(f'plots/{filename15} {parameter}-sensitivity extra Firms.pdf',bbox_inches='tight')

# Welfare
plt.figure(figsize=(6,4))
plt.plot(periods_array,track15[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,1.5),marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track20[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,2.0),marker='^',markevery=5,color='#009E73',linestyle="dashed")
plt.plot(periods_array,track10[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,1.0),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
#plt.plot(periods_array,track05[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,0.5),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
plt.plot(periods_array,track00[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{optimized~tax,}",parameter_tex,0.0),marker='^',markevery=5,color='#009E73',linestyle="dotted")
plt.plot(periods_array,track_static15[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2')
plt.plot(periods_array,track_static20[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,2.0),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
#plt.plot(periods_array,track_static15[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(5,5)))
#plt.plot(periods_array,track_static05[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,0.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(1,6)))
plt.plot(periods_array,track_static00[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{static~tax,}",parameter_tex,0.0),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
plt.plot(periods_array,track_zero15[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00')
plt.plot(periods_array,track_zero20[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,2.0),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
#plt.plot(periods_array,track_zero15[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(5,5)))
#plt.plot(periods_array,track_zero05[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,0.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(1,7)))
plt.plot(periods_array,track_zero00[:,3]/Wnorm,label=r"$\Omega,~\mathrm{}~{}={}$".format("{no~tax,}",parameter_tex,0.0),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
plt.xlabel("Years")
plt.ylabel("Welfare")
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.xlim(0,50)
plt.grid()
plt.savefig(f'plots/{filename15} {parameter}-sensitivity extra Welfare.pdf',bbox_inches='tight')

# Demand for green goods
plt.figure(figsize=(6,4))
plt.plot(periods_array,track15[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,1.5),marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track20[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,2.0),marker='^',markevery=5,color='#009E73',linestyle="dashed")
plt.plot(periods_array,track10[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,1.0),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
#plt.plot(periods_array,track05[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,0.5),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
plt.plot(periods_array,track00[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{optimized~tax,}",parameter_tex,0.0),marker='^',markevery=5,color='#009E73',linestyle="dotted")
plt.plot(periods_array,track_static15[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2')
plt.plot(periods_array,track_static20[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,2.0),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
#plt.plot(periods_array,track_static15[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(5,5)))
#plt.plot(periods_array,track_static05[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,0.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(1,6)))
plt.plot(periods_array,track_static00[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{static~tax,}",parameter_tex,0.0),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
plt.plot(periods_array,track_zero15[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00')
plt.plot(periods_array,track_zero20[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,2.0),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
#plt.plot(periods_array,track_zero15[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(5,5)))
#plt.plot(periods_array,track_zero05[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,0.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(1,7)))
plt.plot(periods_array,track_zero00[:,4]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{y}","{no~tax,}",parameter_tex,0.0),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
plt.xlabel("Years")
plt.ylabel("Demand")
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.xlim(0,50)
plt.grid()
plt.savefig(f'plots/{filename15} {parameter}-sensitivity extra GreenDemand.pdf',bbox_inches='tight')

# Demand for brown goods
plt.figure(figsize=(6,4))
plt.plot(periods_array,track15[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,1.5),marker='^',markevery=5,color='#009E73')
plt.plot(periods_array,track20[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,2.0),marker='^',markevery=5,color='#009E73',linestyle="dashed")
#plt.plot(periods_array,track15[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,1.5),marker='^',markevery=5,color='#009E73',linestyle=(0,(5,5)))
#plt.plot(periods_array,track05[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,0.5),marker='^',markevery=5,color='#009E73',linestyle=(0,(1,5)))
plt.plot(periods_array,track00[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{optimized~tax,}",parameter_tex,0.0),marker='^',markevery=5,color='#009E73',linestyle="dotted")
plt.plot(periods_array,track_static15[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2')
plt.plot(periods_array,track_static20[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,2.0),marker='o',markevery=5,color='#0072B2',linestyle="dashed")
#plt.plot(periods_array,track_static15[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,1.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(5,5)))
#plt.plot(periods_array,track_static05[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,0.5),marker='o',markevery=5,color='#0072B2',linestyle=(0,(1,6)))
plt.plot(periods_array,track_static00[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{static~tax,}",parameter_tex,0.0),marker='o',markevery=5,color='#0072B2',linestyle="dotted")
plt.plot(periods_array,track_zero15[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00')
plt.plot(periods_array,track_zero20[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,2.0),marker='X',markevery=5,color='#D55E00',linestyle="dashed")
#plt.plot(periods_array,track_zero15[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,1.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(5,5)))
#plt.plot(periods_array,track_zero05[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,0.5),marker='X',markevery=5,color='#D55E00',linestyle=(0,(1,7)))
plt.plot(periods_array,track_zero00[:,5]/Ynorm,label=r"$\bar{},~\mathrm{}~{}={}$".format("{Y}","{no~tax,}",parameter_tex,0.0),marker='X',markevery=5,color='#D55E00',linestyle="dotted")
plt.xlabel("Years")
plt.ylabel("Demand")
plt.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.xlim(0,50)
plt.grid()
plt.savefig(f'plots/{filename15} {parameter}-sensitivity extra BrownDemand.pdf',bbox_inches='tight')


