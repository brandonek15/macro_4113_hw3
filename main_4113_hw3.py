import os
import platform
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.optimize
from scipy.interpolate import RectBivariateSpline

#Set the directory
if platform.system() == 'Windows':
    ROOT = 'C:\\Users\\Brand\\OneDrive\\Econ\\Northwestern\\Macro 3\\hw3'
elif platform.system() == 'Linux':
    #Not using linux for this pset so not setting directory
    ROOT = '/home/blz782/Research_Projects/Gift_Cards'


CODE_PATH = os.path.join(ROOT,'code')

FINAL_OUTPUT_PATH = os.path.join(ROOT,"output")

DIRECTORY_LIST = [FINAL_OUTPUT_PATH,CODE_PATH]

for path in DIRECTORY_LIST:
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory " + path + " failed")
    else:
        print ("Successfully created the directory " + path)

#Parameters passed in by user
R_GRID_WIDTH = (1/200)
ALPHA = .36
DELTA = .025
BETA = .99
TOLERANCE = 1*10e-2
IND_K_MAX = 100
IND_K_MIN = 0
IND_K_GRID_N = 101
AGG_K_MAX = 80
AGG_K_MIN = 10
AGG_K_GRID_N = 8
N_SIMULATION = 1000
T_SIMULATION = 1100

#Make the PI Transition matrix
# PI = PI[z,z',s,s'] give probability of being in state z',s' given I am in z,s (where zero is low, 1 is high)
PI = np.zeros((2,2,2,2))
PI[0,0,0,0] = 21/40
PI[0,0,0,1] = 7/20
PI[0,1,0,0] = 1/32
PI[0,1,0,1] = 3/32
PI[0,0,1,0] = .0389
PI[0,0,1,1] = .8361
PI[0,1,1,0] = .0021
PI[0,1,1,1] = .1229
PI[1,0,0,0] = 3/32
PI[1,0,0,1] = 1/32
PI[1,1,0,0] = 7/24
PI[1,1,0,1] = 7/12
PI[1,0,1,0] = .0091
PI[1,0,1,1] = .1159
PI[1,1,1,0] = .0243
PI[1,1,1,1] = .8507

#Aggregate Transition Matrix (AGG_PI[s,s'] = probability of going from s to s', s=0 is bad, s=1 is good
AGG_PI = np.zeros((2,2))
AGG_PI[0,0] = 7/8
AGG_PI[0,1] = 1/8
AGG_PI[1,0] = 1/8
AGG_PI[1,1] = 7/8

def main():
    print("Starting main_pset3.py")
    # Set seed so I get the same results every time
    np.random.seed(1)

    #Define the capital grids
    ind_K_grid = np.arange(start=0, stop=.5+.005, step=.005)
    ind_K_grid = IND_K_MIN + ind_K_grid**7/((.5)**7)*(IND_K_MAX-IND_K_MIN)
    agg_K_grid = np.arange(start=AGG_K_MIN,stop = AGG_K_MAX+10,step = 10)
    #Define individual state (labor =1,0) and aggregate state (z=.99,1.01)
    labor_states = [0,1]
    productivity_states = [.99,1.01]

    alpha_b,beta_b,alpha_g,beta_g = [0,1,0,1]
    #Initial guess for Law of motion
    def LOM(ln_K,agg_productivity):
        if agg_productivity ==0.99:
            return alpha_b + beta_b*ln_K
        if agg_productivity == 1.01:
            return alpha_g + beta_g*ln_K

    #Get optimal policy function k'[k,K,s,z] (part a)
    kprime = solve_policy_function(ind_K_grid,agg_K_grid,LOM,labor_states,productivity_states)

    toview = kprime[:,3,:,1]

    # Create a random vector from uniform distribution
    rand = np.random.uniform(size=N_SIMULATION)

    #Set the aggregate state [z] to be the good state, start at stationary distribution (4% unemployed)
    z = 1
    # For  each individual, store assets (column 0), labor state (column 1)
    simulated_individuals = np.zeros((N_SIMULATION,2)).astype(int)
    # First column is where on asset grid
    #Set everyone's capital to steady state capital in good state (derived from Euler Equation)
    #(1/css) = (1/css)beta(r_ss+1-delta), r_ss = z*alpha*(Lss/Kss)^(1-alpha)
    ind_K_ss = np.power((1/BETA-(1-DELTA))/(productivity_states[1]*ALPHA*np.power(.96,1-ALPHA)),1/(ALPHA-1))
    simulated_individuals[:,0] = ind_K_ss
    simulated_individuals[:,1] = (np.where(rand < .04, 0, 1))

    for t in range(T_SIMULATION):
        #First decide whether the aggreind_K_ssgate productivity transitions
        #If the random number is less than pi[0,0] =7/8 or pi[1,0] = 1/8, then state tomorrow is 0
        if np.random.uniform() < AGG_PI[z,0]:
            z_prime = 0
        else:
            z_prime = 1

        #Replace this part with index of agg_K_grid that is closest to aggregate K. Agents only know
        #the policy function at given K's in our grid but actual K may be different
        agg_K = np.average(ind_K_grid[simulated_individuals[:,0]])
        agg_K_index = (np.abs(agg_K_grid - agg_K)).argmin()

        #Back out corresponding index for policy function. Currently policy function gives optimal capital for tommorrow
        # , but need the index on the capital grid corresponding to optimal capital
        ind_index = np.argmin(np.abs(np.subtract.outer(simulated_individuals[:,0], ind_K_grid)), axis=1)

        #Calculate assets tomorrow based on policy function,current assets[:],Aggregate capital [Kbar],agg_state (z)
        # Assets (first column) = k'[:,0] (vector) = (1-l) (vector)*k'[:,l=0] (vector)  + (l)*k'[:,l=1]
        simulated_individuals[:, 0] = (1 - simulated_individuals[:, 1]) * kprime[ind_index,agg_K_index,0,z] \
                + simulated_individuals[:, 1] * kprime[ind_index,agg_K_index,1,z]

        rand = np.random.uniform(size=N_SIMULATION)
        # Labor (2nd column) tomorrow is a random function = (1-l)(p00 *0+ p01*1) + (l)(p10*0+p11*1),p## conditional on z,z'
        simulated_individuals[:, 1] = (1 - simulated_individuals[:, 1]) * \
            (np.where(rand < (PI[z,z_prime,0,0]/(PI[z,z_prime,0,0]+PI[z,z_prime,0,1])), 0, 1)) \
            + simulated_individuals[:, 1] * np.where(rand < (PI[z,z_prime,1,0]/(PI[z,z_prime,1,0]+PI[z,z_prime,1,1])), 0, 1)
        print("Value of Agg K is " + str(round(agg_K,2)))

    #todo I ran one simulation. Now do the updating with aggregate belief using the regression
    #also modularize the code so the simulation part is seperate code
    print("Finishing")

def solve_policy_function(ind_K_grid,agg_K_grid,law_of_motion,labor_states,productivity_states):
    '''This is part (a) of the homework. It take a law of motion for capital (function)
    and solves for the policy function assuming this LOM is true'''
    #To use bicubic spline, need the policy function to be of dim (N_k,N_K)
    kprime_new,kprime_prime,cprime,c= [np.zeros((IND_K_GRID_N,AGG_K_GRID_N,2,2)),  \
        np.zeros((IND_K_GRID_N,AGG_K_GRID_N,2,2)), \
        np.zeros((IND_K_GRID_N,AGG_K_GRID_N,2,2)), np.zeros((IND_K_GRID_N,AGG_K_GRID_N,2,2))]
    
    #Set initial guess of policy function  k'(k,K,s,z) = .9k
    for s in range(len(labor_states)):
        for z in range(len(productivity_states)):
            for K in range(AGG_K_GRID_N):
                kprime_new[:,K,s,z] = 0.9*ind_K_grid

    #This will look at the max deviation across the entire 4 dimensional grid of k,K,s,z
    norm = 100

    while norm > TOLERANCE:
        #Reset norm to zero (because I am taking maxes)
        norm = 0
        kprime_old = kprime_new.copy()
        #For a fixed s,z, use a bicubic spline to get k''[k,K]
        for s in range(len(labor_states)):
            for z in range(len(productivity_states)):
                #use a bicubic spline on k'[k,K] to get k'(k,K) (so it is a function that I input k,K)
                cs_of_kprime = RectBivariateSpline(ind_K_grid,agg_K_grid,kprime_old[:,:,s,z])
                for K in range(AGG_K_GRID_N):
                    #LOM give value of ln(K'), so need to exponentiate
                    Kprime = np.exp(law_of_motion(ln_K=np.log(agg_K_grid[K]),agg_productivity=productivity_states[z]))
                    #For each K_n , Get k''[k,K_n] = k'(k'[k,K_n],K'[K_n]), note K'[K_n] is from LOM
                    kprime_prime[:,K,s,z] = cs_of_kprime(kprime_old[:,K,s,z], Kprime).flatten()
                    #Back out c' from budget constraint (c' = l'*w' +(r'+1-delta)k' - k'')
                    cprime[:,K,s,z] = labor_states[s]*get_wage(productivity_states[z],Kprime)+kprime_old[:,K,s,z]* \
                             (get_rental_rate(productivity_states[z],Kprime)+1-DELTA)-kprime_prime[:,K,s,z]

        #For each s_bar,z_bar,K_bar...
        for s in range(len(labor_states)):
            for z in range(len(productivity_states)):
                for K in range(AGG_K_GRID_N):
                    #Calculate Kprime
                    Kprime = np.exp(law_of_motion(ln_K=np.log(agg_K_grid[K]),agg_productivity=productivity_states[z]))

                    # Back out c[k,K_bar,s_bar,z_bar] from Euler equation (1/c) = E[(1/c')*Beta*(r+1-delta)]
                    expectation_term = 0

                    #Now loop over s' and z' to make computing the expectation term easier
                    for s_prime in range(len(labor_states)):
                        for z_prime in range(len(productivity_states)):
                            expectation_term += PI[z,z_prime,s,s_prime]*(1/cprime[:,K,s,z])*BETA* \
                                (get_rental_rate(productivity_states[z],Kprime)+1-DELTA)

                    #c[k,K_bar,s_bar,z_bar] = 1/ E[(1/c')*Beta*(r+1-delta)]
                    c[:,K,s,z] = 1/expectation_term

                    # Back out k' from budget constraint (k' = l*w +(r+1-delta)k - c)
                    kprime_new[:, K, s, z] = labor_states[s] * get_wage(productivity_states[z], Kprime) + \
                        ind_K_grid*(get_rental_rate(productivity_states[z], Kprime) + 1 - DELTA) - c[:,K,s,z]

                    #Compare kprime_new to kprime old. If close enough, we have our policy function
                    norm = max(np.linalg.norm(kprime_new[:,K,s,z] - kprime_old[:,K,s,z], ord=np.inf),norm)

        if norm > TOLERANCE:
            # Take an average of the new kprime and the old prime as described in the HW
            kprime_new = .4 * kprime_new + .6 * kprime_old

        print("the value of the norm is " + str.format('{0:.10f}', norm))
    return kprime_new

def get_wage(agg_productivity,agg_capital):
    '''Returns wage as a fucntion of aggregate productiviy and capital (from firm profit max)
    Note that aggregate labor is pinned down by aggregate productivity'''
    if agg_productivity == .99:
        return (1-ALPHA)*agg_productivity*np.power(agg_capital/.9,ALPHA)
    elif agg_productivity == 1.01:
        return (1-ALPHA)*agg_productivity*np.power(agg_capital/.96,ALPHA)

def get_rental_rate(agg_productivity, agg_capital):
    '''Returns rental rate of capital as a fucntion of aggregate productiviy and capital
    (from firm profit max)'''
    if agg_productivity == .99:
        return ALPHA * agg_productivity * np.power(.9/agg_capital, 1-ALPHA)
    elif agg_productivity == 1.01:
        return (1 - ALPHA) * agg_productivity * np.power(.96/agg_capital, 1-ALPHA)

def policy_function_method_2(capital_grid):

    #Method 2 (h-i)

    #Guess k'[k] (tomorrows capital)
    kprime_new = capital_grid
    norm = 100

    while norm > TOLERANCE:
        kprime_old = kprime_new.copy()
        #Use cubic spline on k'[k] to get k'(k)
        cs_of_kprime = CubicSpline(capital_grid, kprime_old)
        #Get k''[k] = k'(k'[k])
        kprime_prime = cs_of_kprime(kprime_old)
        #Get c'[k] = (1-delta)*k'[k]+Ak'[k]^alpha - k''[k]
        cprime = (1-DELTA)*kprime_old+A*np.power(kprime_old,ALPHA)-kprime_prime
        #Get c[k] from euler. c[k] = c'[k]/(beta*(1-delta+A*alpha*k'[k]^(alpha-1))
        c = cprime/(BETA*(1-DELTA + A*ALPHA*np.power(kprime_old,ALPHA-1)))
        #Get k_new'[k] = (1-delta)k0 + A*k0^alpha - c[k]
        kprime_new = (1-DELTA)*capital_grid + A*np.power(capital_grid,ALPHA)-c
        #Don't allow the capital tomorrow to be  negative
        kprime_new[kprime_new <0] = 10e-3
        #Computer norm of ||k_new'[k]-k'[k]|| and see if within tolerance
        norm = np.linalg.norm(kprime_new - kprime_old, ord=np.inf)

    #g[k] = k'[k]
    policy_function = kprime_new

if __name__ == '__main__':
    main()