import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

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


""" Number of Extrema Check """

def check():
    # Checks if value function V has exactly 1 extremum so that the optimization result is valid
    
    # Load data
    R = np.load(f'data/{filename} R.npy')
    T = np.load(f'data/{filename} T.npy')
    
    # Optimization variable to test
    resolution = 200 # resolution of test
    T_array = np.linspace(0, 20, resolution) # 20 is the maximum tax tested
    
    for period in range(periods):
        print(f"Check period {period}.")
        for i in range(gridsize):                       # every μ
            for j in range(gridsize):                   # every γ
        
                V_array = np.zeros(resolution) # the negative value function values for different tax rates T at period period
                for k in range(resolution):
                    V_array[k] = V(T_array[k],μ_array[i],γ_array[j],period,R)
                                    
                T_opt = T_array[np.argmin(V_array)]
                V_opt = np.min(V_array)
                T_stored = T[period,i,j]
                V_stored = V(T_stored,μ_array[i],γ_array[j],period,R)
            
                
                #print(T_opt,T[period,μ,γ])
                
                if abs((T_opt-T_stored)/(T_opt+0.001)) > 0.03 and abs((V_opt-V_stored)/(V_opt+0.001)) > 0.005: 
                    # it is checked if there is a difference in T_opt
                    # 3% can be lowered, but with a resolution of 200, one step is already 0.1 which is 10% if the tax is 1
                    # then, it is confirmed that this difference has an impact on the value function of more than 0.5% (can also be lowered)
                    
                    print(f"Difference in optimal tax rate detected. μ={np.round(μ_array[i],2)},γ={np.round(γ_array[j],2)},period={period}.")
                    print(f"Optimal T in array: {np.round(T_opt,2)}, optimal stored T: {np.round(T[period,i,j],2)}, deviation: {np.round(100*abs((T_opt-T[period,i,j])/(T_opt+0.01)),2)}%")
                    plt.figure()
                    plt.plot(T_array,-V_array)
                    plt.axvline(x=T_opt,linestyle='--')
                    plt.axvline(x=T_stored,linestyle=':')
                    plt.xlabel("Tax T")
                    plt.ylabel("Value function V")
                    plt.grid()
                            
    print("Check completed for all periods.")



""" Run check for all parameter sets """

print("Checking that the optimal tax rate has been found.")
# Check base case
parameters = parameters_orig.copy()
initialization()
filename_orig = filename
print("Check base case.")
check()

"""
# Check sensitivity parameter sets
for i in range(11):
    for j in ["up","down"]:
        print(f"Check for {j}-deviation in {parameters_unicode[i]}.")
        parameters[i] = globals()[parameters_unicode[i]+"_"+j]                  # before it is parameters = parameters_orig, change of one parameter, e.g. to σ_up
        initialization()                                                        # sets the global parameter variables according to the array parameters
        check()                           
        parameters = parameters_orig.copy()                                     # reset of the array parameters to the base case
"""





