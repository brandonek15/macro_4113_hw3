import os
import platform
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd
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
N_SIMULATION = 10000
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

    #Define the capital grids
    ind_K_grid = np.arange(start=0, stop=.5+.005, step=.005)
    ind_K_grid = IND_K_MIN + ind_K_grid**7/((.5)**7)*(IND_K_MAX-IND_K_MIN)
    agg_K_grid = np.arange(start=AGG_K_MIN,stop = AGG_K_MAX+10,step = 10)
    #Define individual state (labor =1,0) and aggregate state (z=.99,1.01)
    labor_states = [0,1]
    productivity_states = [.99,1.01]

    #Initial guess for Law of motion
    alpha_b_new,beta_b_new,alpha_g_new,beta_g_new = [0,1,0,1]

    #Type 1 is parts b,c. Type 2 is part d (robustness). All I am doing is chaning the seed!
    #Note when doing type 2, the initial guess for LOM is the last one from step 1
    for type in range(1,3):
        if type == 1:
            title = "Original"
            save_file = "original_agg_capital.png"
        elif type == 2:
            title = "Robust"
            save_file = "robust_agg_capital.png"

        norm = 100
        while norm >1*10e-5:
            #Update coefficients on LOM
            alpha_b_old,beta_b_old,alpha_g_old,beta_g_old = [alpha_b_new,beta_b_new,alpha_g_new,beta_g_new]

            def LOM(ln_K,agg_productivity):
                if agg_productivity ==0.99:
                    return alpha_b_old + beta_b_old*ln_K
                if agg_productivity == 1.01:
                    return alpha_g_old + beta_g_old*ln_K

            #Get optimal policy function k'[k,K,s,z] (part a)
            kprime = solve_policy_function(ind_K_grid,agg_K_grid,LOM,labor_states,productivity_states)

            #Get the aggregate time series with the given policy function kprime
            agg_K_time_series = simulate_economy(ind_K_grid,agg_K_grid,productivity_states,kprime,seed=type)

            #Get new coefficients for LOM based on this iteration (and R^2)
            alpha_b_new,beta_b_new,r2_b,alpha_g_new,beta_g_new,r2_g = get_autoregressive_coef(agg_K_time_series)

            #See if LOM coefficients have changed by a lot or not
            norm =max(abs(alpha_b_new-alpha_b_old),abs(beta_b_new-beta_b_old), \
                      abs(alpha_g_new-alpha_g_old),abs(beta_g_new-beta_g_old))

        #part (c) Plot aggregate capital stock time series in the lsat simulation vs
        #The aggregate capital stock using initial capital and LOM
        plot_actual_vs_predicted_capital(agg_K_time_series,LOM,productivity_states \
            ,r2_b,r2_g,title,save_file)

    print("Finishing")

def plot_actual_vs_predicted_capital(agg_K_time_series,LOM,productivity_states,r2_b,r2_g,title,file_name):
    '''This plots the actual series of capital from the last iteration from the
    approximation using the LOMs for good and bad states.'''
    time_grid = np.arange(start=0,stop=T_SIMULATION-100)
    df = pd.DataFrame(agg_K_time_series, columns=['K', 'agg_state'])
    # Drop first  100 periods
    df = df.iloc[100:]
    #Convert to int to use as an index
    df = df.astype({'agg_state': 'int32'})
    #Create the approximation from the law of motion
    df['K_from_LOM'] = 0
    df['K_from_LOM'].iloc[0] = df['K'].iloc[0]
    for t in range(1,T_SIMULATION-100):
        #Use LOM. LOM function gives ln_K_t+1 so need to exponentiate
        df['K_from_LOM'].iloc[t] = np.exp(LOM(np.log(df['K_from_LOM'].iloc[t-1]), \
                            productivity_states[df['agg_state'].iloc[t-1]]))

    plt.plot(time_grid,df['K'],label='Actual K',color='blue')
    plt.plot(time_grid,df['K_from_LOM'], label='K from Law of Motion', color='green')
    plt.xlabel('Period (t)')
    plt.ylabel('Aggregate Capital (K)')
    plt.title(title + ' Capital Graph. R2 for good: ' + str(round(r2_g,3)) + ' R2 for bad: ' + str(round(r2_b,3)))
    plt.legend(loc='best')

    #Save plot
    file_name = os.path.join(FINAL_OUTPUT_PATH, file_name)
    plt.savefig(file_name, dpi=150)
    #plt.show()
    plt.close()

def simulate_economy(ind_K_grid,agg_K_grid,productivity_states,kprime,seed):
    '''This program simulates the shocks for the individuals and returns aggregate capital after each t'''
    # Set seed so I get the same set of shocks every time
    np.random.seed(seed)
    # Create a random vector for each t from uniform distribution. This way each iteration has the same shock
    rand = np.random.rand(N_SIMULATION,T_SIMULATION)

    #Set the aggregate state [z] to be the good state, start at stationary distribution (4% unemployed)
    z_prime = 1
    # For  each individual, store assets (column 0), labor state (column 1)
    simulated_individuals = np.zeros((N_SIMULATION,2))
    # First column is where on asset grid
    #Set everyone's capital to steady state capital in good state (derived from Euler Equation)
    #(1/css) = (1/css)beta(r_ss+1-delta), r_ss = z*alpha*(Lss/Kss)^(1-alpha)
    ind_K_ss = np.power((1/BETA-(1-DELTA))/(productivity_states[1]*ALPHA*np.power(.96,1-ALPHA)),1/(ALPHA-1))
    simulated_individuals[:,0] = ind_K_ss
    simulated_individuals[:,1] = (np.where(rand[:,0] < .04, 0, 1))
    #Create an array that stores aggregate K (the object of interest) (column 0), and state (column 1)
    agg_K = np.zeros((T_SIMULATION,2))

    for t in range(T_SIMULATION):
        #Update todays state with yesterdays tomorrow state
        z = z_prime
        #Store the time series
        agg_K[t,1] = z
        #First decide whether the aggreind_K_ssgate productivity transitions
        #If the random number is less than pi[0,0] =7/8 or pi[1,0] = 1/8, then state tomorrow is 0
        if rand[0,t] < AGG_PI[z,0]:
            z_prime = 0
        else:
            z_prime = 1

        #Replace this part with index of agg_K_grid that is closest to aggregate K. Agents only know
        #the policy function at given K's in our grid but actual K may be different
        agg_K[t,0] = np.average(simulated_individuals[:,0])
        agg_K_index = (np.abs(agg_K_grid - agg_K[t,0])).argmin()

        #Back out corresponding index for policy function. Currently policy function gives optimal capital for tommorrow
        # , but need the index on the capital grid corresponding to optimal capital
        ind_index = np.argmin(np.abs(np.subtract.outer(simulated_individuals[:,0], ind_K_grid)), axis=1)

        #Calculate assets tomorrow based on policy function,current assets[:],Aggregate capital [Kbar],agg_state (z)
        # Assets (first column) = k'[:,0] (vector) = (1-l) (vector)*k'[:,l=0] (vector)  + (l)*k'[:,l=1]
        simulated_individuals[:, 0] = (1 - simulated_individuals[:, 1]) * kprime[ind_index,agg_K_index,0,z] \
                + simulated_individuals[:, 1] * kprime[ind_index,agg_K_index,1,z]

        # Labor (2nd column) tomorrow is a random function = (1-l)(p00 *0+ p01*1) + (l)(p10*0+p11*1),p## conditional on z,z'
        simulated_individuals[:, 1] = (1 - simulated_individuals[:, 1]) * \
            (np.where(rand[:,t] < (PI[z,z_prime,0,0]/(PI[z,z_prime,0,0]+PI[z,z_prime,0,1])), 0, 1)) \
            + simulated_individuals[:, 1] * np.where(rand[:,t] < (PI[z,z_prime,1,0]/(PI[z,z_prime,1,0]+PI[z,z_prime,1,1])), 0, 1)
        #print("Value of Agg K is " + str(round(agg_K,2)))
    return agg_K

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

def get_autoregressive_coef(agg_K_time_series):
    '''This program runs the autoregressions to update the new coefficients on the LOM
    Returns the alpha_b,beta_b,alpha_g,beta_g'''
    df = pd.DataFrame(agg_K_time_series, columns=['K', 'agg_state'])
    df['L1_K'] = df['K'].shift(1)
    df['L1_agg_state'] = df['agg_state'].shift(1)
    #Set lagged K to be missing if it changed states in that period  (not sure if this should be there)
    #df.loc[df['L1_agg_state']-df['agg_state']!=0, 'L1_K'] = np.nan
    #Create log variables
    df['ln_K'] = np.log(df['K'])
    df['L1_ln_K'] = np.log(df['L1_K'])
    #Drop first  100 periods
    df = df.iloc[100:]
    #Now can subset to get time series for good to good states, and bad to bad states
    good_state_df = df[df['agg_state']==1]
    bad_state_df = df[df['agg_state']==0]
    good_results = smf.ols('ln_K ~ L1_ln_K', data=good_state_df).fit()
    bad_results = smf.ols('ln_K ~ L1_ln_K', data=bad_state_df).fit()
    #Extract parameters from model
    alpha_g = good_results.params[0]
    beta_g = good_results.params[1]
    r2_g = good_results.rsquared

    alpha_b = bad_results.params[0]
    beta_b = bad_results.params[1]
    r2_b = bad_results.rsquared

    return alpha_b,beta_b,r2_b,alpha_g,beta_g,r2_g

if __name__ == '__main__':
    main()