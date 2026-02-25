import numpy as np

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

periods_array = np.arange(periods)      # from 0 to periods-1 



""" Model functions """

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



""" Parameter Initialization for Changed Parameters """

def initialization():                    # sets the parameter variables according to the array "parameters"
    
    global σ,g,χ,ζ,m,λ,β,d,e,μ0,γ0,t,T_static,track_d0,track,track_zero,Tnorm,Wnorm
    #print(parameters)
    
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
    
    t = -σ*(χ+ζ)                            # green tax is fixed to the static green tax
    T_static = (1-σ)*λ-σ*χ                  # static brown tax

    
    #print(σ,g,χ,ζ,m,λ,β,d,e,μ0,γ0)
    filename = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d{d} e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}' 
    filename_d0 = f'σ{σ} g{g} χ{χ} ζ{ζ} m{m} λ{λ} β{β} d0.0 e{e} μ{μ0} γ{γ0} I{I} gridsize{gridsize} periods{periods}' 
    
    #print(filename)
    track_d0 = np.load(f'data/{filename_d0} Track.npy')  # the optimization results for d = 0
    track = np.load(f'data/{filename} Track.npy')                 
    track_zero = np.load(f'data/{filename} Track_zero.npy')
    
    track_static = np.load(f'data/{filename} Track_static.npy') # for plotting
    Tnorm = track_static[0,2]                                   # for plotting
    Wnorm = track_static[0,3]                                   # for plotting



""" Tracking the development for d=1.5 when the optimized tax for d=0 (ignoring changing preferences) is imposed """

def tracking_ignore():
    track_ignore = np.zeros((periods,4))       # axis 1: [0]: μ, [1]: γ, [2]: T, [3]: Ω
    period = 0
    μ = μ0
    γ = γ0
    
    while period < periods:
        track_ignore[period,0] = μ
        track_ignore[period,1] = γ
        track_ignore[period,2] = track_d0[period,2]
        track_ignore[period,3] = -Ω(track_ignore[period,2],μ,γ)
        μ = μ_(track_ignore[period,2],μ,γ)
        γ = γ_(track_ignore[period,2],μ,γ)
        period += 1
        
    return track_ignore



""" Calculate the welfare differences """

def welfare_diff():
    welfare = 0          # welfare sum with optimized tax normalized by the no tax case
    welfare_ignore = 0   # welfare sum with optimized tax for d = 0 normalized by the no tax case
    
    for period in range(50):
        welfare = welfare + (track[period,3]-track_zero[period,3])*β**period
        welfare_ignore = welfare_ignore + (track_ignore[period,3]-track_zero[period,3])*β**period
        
    print(round((welfare_ignore-welfare)/welfare*100,1),"%")


""" Loop over all parameter changes """

print("Welfare loss by ignoring changing preferences when optimizing the tax")

print("Base case:")
parameters = parameters_orig.copy()
initialization()                        # imports the trajectories from the optimization 
track_ignore = tracking_ignore()        # calculates the trajectory when applying the tax optimized for d=0 in the d=1.5 system
welfare_diff()                          # calculates the welfare loss
#plots()


for i in range(11):                                                             # i is the index of the changed parameter

    print(f"Up-deviation in {parameters_unicode[i]}:")
    parameters = parameters_orig.copy()
    parameters[i] = globals()[parameters_unicode[i]+"_up"]                      # change of one parameter, e.g. to σ_up
    initialization()                                                      
    track_ignore = tracking_ignore()
    welfare_diff()
    
    print(f"Down-deviation in {parameters_unicode[i]}:")
    parameters = parameters_orig.copy()
    parameters[i] = globals()[parameters_unicode[i]+"_down"]                      
    initialization()
    track_ignore = tracking_ignore()                                                       
    welfare_diff()









