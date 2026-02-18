import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interpn


""" Set parameters """

σ = 0.5                 # substitution elasticitiy 
g = 0.5                 # preference shift 
χ = 3.0                   # marginal brown cost
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



""" Parameter Initialization """

parameters_orig = np.array([σ,g,χ,ζ,m,λ,β,d,e,μ0,γ0])       # base case parameters not to be changed
parameters_unicode = ["σ","g","χ","ζ","m","λ","β","d","e","μ0","γ0"]
parameters_tex = ["\sigma","g","\chi","\zeta","m","\lambda",r"\beta","d","e","\mu_0","\gamma_0"]

periods_array = np.arange(periods)      # from 0 to periods-1        
μ_array = np.linspace(0,1,gridsize)     # all μ values 
γ_array = np.linspace(0,1,gridsize)     # all γ values


def initialization():                   # sets the parameter variables according to the array "parameters"
    
    global σ,g,χ,ζ,m,λ,β,d,e,μ0,γ0,filename,t,T_static
    
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

    t = -σ*(χ+ζ)                    # green tax is fixed to the static green tax
    T_static = (1-σ)*λ-σ*χ          # static brown tax



""" Model functions """

def Ω(T,μ,γ):                           # negative welfare function (to use minimization instead of maximization)            
    return -(γ*(1+μ*g)*w() + (1-γ)*(1-μ*g)*W(T) + I - γ**2 * m/2)

def w():                                # part of the welfare function 
    return κ(ζ+t)/(1-σ) - (χ+ζ) * κ(ζ+t)**(1/(1-σ))

def W(T):                               # part of the welfare function 
    return κ(T)/(1-σ) -  (χ+λ) * κ(T)**(1/(1-σ))

def V(T,μ,γ,period,R):                  # negative welfare of the current and all future periods
    return Ω(T,μ,γ) - β * interpn((μ_array,γ_array),R[period+1,:,:],(μ_(T,μ,γ),γ_(T,μ,γ)),method='linear').item()

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



""" Optimization using backward induction """

def optimization():
    
    R = np.zeros((periods,gridsize,gridsize))       # welfare of current and all future periods with optimal brown tax, given period, μ, and γ
    T = np.zeros((periods,gridsize,gridsize))       # optimal brown tax T, given period, μ, and γ
    
    period = periods-1                              # last period: optimize Ω instead of V
    print("Calculating period",period)
    for i in range(gridsize):                       # every μ
        for j in range(gridsize):                   # every γ
            res = minimize(Ω, x0=(T_static), args=(μ_array[i],γ_array[j]), bounds=[(0,20)], tol=1e-12) #tol=1e-10 
            R[period,i,j] = -res.fun                # optimal welfare given μ, γ
            T[period,i,j] = res.x.item()            # optimal T given μ, γ
    period -= 1
    
    while period >= 0:                   
        print("Calculating period",period)
        for i in range(gridsize):                   # every μ
            for j in range(gridsize):               # every γ
                res = minimize(V, x0=(T[period+1,i,j]), args=(μ_array[i],γ_array[j],period,R), bounds=[(0,20)], tol=1e-12) 
                R[period,i,j] = -res.fun            # optimal welfare of current and future periods given μ, γ
                T[period,i,j] = res.x.item()        # optimal T given μ, γ
        period -= 1
        
    np.save(f'data/{filename} R',R)
    
    return T



""" Forward tracking """

def tracking():
    
    global t
    
    """ Optimized tax """
    
    track = np.zeros((periods,6))       # axis 1: [0]: μ, [1]: γ, [2]: T, [3]: Ω, [4]: y, [5]: Y
    period = 0
    μ = μ0
    γ = γ0

    while period < periods:
        track[period,0] = μ
        track[period,1] = γ
        track[period,2] = interpn((μ_array,γ_array),T[period,:,:],(μ,γ),method='linear')
        track[period,3] = -Ω(track[period,2],μ,γ)
        track[period,4] = y(μ,γ)
        track[period,5] = Y(track[period,2],μ,γ)
        μ = μ_(track[period,2],μ,γ)
        γ = γ_(track[period,2],μ,γ)
        period += 1
    
    np.save(f'data/{filename} Track',track)
    
    
    """ Static tax """
    
    track = np.zeros((periods,6))     
    period = 0
    μ = μ0
    γ = γ0

    while period < periods:
        track[period,0] = μ
        track[period,1] = γ
        track[period,2] = T_static
        track[period,3] = -Ω(track[period,2],μ,γ)
        track[period,4] = y(μ,γ)
        track[period,5] = Y(track[period,2],μ,γ)
        μ = μ_(track[period,2],μ,γ)
        γ = γ_(track[period,2],μ,γ)
        period += 1
    
    np.save(f'data/{filename} Track_static',track)
    
    
    """ No tax """
    
    t_temp = t
    t = 0
    track = np.zeros((periods,6))       
    period = 0
    μ = μ0
    γ = γ0

    while period < periods:
        track[period,0] = μ
        track[period,1] = γ
        track[period,2] = 0
        track[period,3] = -Ω(0,μ,γ)
        track[period,4] = y(μ,γ)
        track[period,5] = Y(track[period,2],μ,γ)
        μ = μ_(0,μ,γ)
        γ = γ_(0,μ,γ)
        period += 1
    
    np.save(f'data/{filename} Track_zero',track)
    
    t = t_temp
    
    

""" Run Optimization """

# Optimization for the base case

print("Optimization for the base case.")
parameters = parameters_orig.copy()
initialization()
filename_orig = filename
T = optimization()
tracking()


# Optimization for sensitivity analysis
for i in range(11):
    for j in ["up","down"]:
        print(f"Optimization for {j}-deviation in {parameters_unicode[i]}.")
        parameters[i] = globals()[parameters_unicode[i]+"_"+j]                  # before it is parameters = parameters_orig, change of one parameter, e.g. to σ_up
        initialization()                                                        # sets the global parameter variables according to the array parameters
        T = optimization()                 
        tracking()                                     
        parameters = parameters_orig.copy()                                     # reset of the array parameters to the base case
        

