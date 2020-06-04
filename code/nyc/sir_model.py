import numpy as np
from numpy.linalg import inv
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import sys
import math

def load_set(path):

    cols=["date","cases","hospitalised", "deaths"]
    df = pd.read_csv(path, sep=",", header=1, names=cols)

    return df

def tilda_functions(I, R, alpha, w, P, t):
    I_tilda = (1/alpha)*I
    R_tilda = (1/alpha)*R
    S_tilda = np.zeros(len(t))
    for i in range(0, len(t)):
        S_tilda[i] = (w/alpha)*P - I_tilda[i] - R_tilda[i]

    return S_tilda, I_tilda, R_tilda

def compute_phi(S, I, alpha, t):
    phi = np.zeros((len(t), 3, 3))
    #phi.shape=((t.shape, 3, 3))
    for i in range(0, len(t)):
        phi[i][0][0] = (S[i]*I[i])/(S[i] + I[i])
        phi[i][0][1] = -1*I[i]
        phi[i][0][2] = (-1/alpha)*I[i]
        phi[i][1][0] = 0
        phi[i][1][1] = I[i]
        phi[i][1][2] = 0
        phi[i][2][0] = 0
        phi[i][2][1] = 0
        phi[i][2][2] = I[i]
    
    return phi


def compute_delta(phi: np.array, beta, gamma):
    column = np.asarray([beta, gamma, 0])
    column.shape = (3, 1)

    delta = phi.dot(column)

    #delta is a 3x1 matrix
    return delta

def compute_delta_v2(I_tilda, R_tilda, t):
    delta = np.zeros((len(t), 3, 1))
    for i in range(1, len(t)):
        delta[i-1][0][0] = I_tilda[i] - I_tilda[i-1]
        delta[i-1][1][0] = R_tilda[i] - R_tilda[i-1]
        delta[i-1][2][0] = 0
    
    return delta


def compute_delta_bar(delta: np.array, theta, ro):
    delta_bar = np.zeros((theta, 3, 1))
    for i in range(0, theta):
        delta_bar[i] = (ro**(theta - i))*delta[i]
    
    #delta_bar is a theta x 3 matrix 
    return delta_bar

def compute_phi_bar(phi: np.array, theta, ro):
    phi_bar = np.zeros((theta, 3, 3))
    for i in range(0, theta):
        phi_bar[i] = (ro**(theta - i))*phi[i]
    
    return phi_bar

def italianGS(I_tilda, R_tilda, alpha, ro, P, t):
    e = 9999999999999
    w_arr = np.arange(0, 1, 1e-6)
    alpha_arr = np.arange(1, alpha, 1e-6)
    parameters_array = np.zeros((3, 1))
    w_star = 0
    alpha_star = 0
    beta_star = 0
    gamma_star = 0

    for i in w_arr:
        for j in alpha_arr:
            S_tilda, I_tilda_n, R_tilda_n = tilda_functions(I_tilda, R_tilda, j, i, P, t)
            
            delta = compute_delta_v2(I_tilda_n, R_tilda_n, t)
            phi = compute_phi(S_tilda, I_tilda_n, j, t)
            delta_bar = compute_delta_bar(delta, len(t), ro)
            phi_bar = compute_phi_bar(phi, len(t), ro)

            parameters_array = inv(phi_bar).dot(delta_bar)
            mse = (delta_bar - phi_bar.dot(parameters_array)).mean(axis=None)
            if mse < e:
                e = mse
                w_star = i
                alpha_star = j
                beta_star = parameters_array[0]
                gamma_star = parameters_array[1]

    return w_star, alpha_star, beta_star, gamma_star







if __name__ == "__main__":

    
    ##############################################
    # Loading set
    ##############################################

    #path = Path("sets\\nyc\\case-hosp-death.csv")
    path = os.path.dirname(__file__) + "\\sets\\nyc\\case-hosp-death.csv"
    set = load_set(path)

    ##############################################
    # Calculating total number of infected people
    # - cases_no array contains total number of infected people
    # up until i-th day
    # - plotting the infected people curve and printing total number
    ##############################################

    days_no = set["date"].size
    cases = pd.DataFrame(set, columns=["cases"])
    cases_arr = cases.values 
    cases_no = np.zeros(cases_arr.size)
    total_cases = 0

    for i in range (0, cases_arr.size):
        total_cases += cases_arr[i]
        cases_no[i] = total_cases
        print(total_cases, i)

    days = np.linspace(0, days_no, days_no)

    #plt.figure()
    #plt.plot(days, cases_no)
    #plt.show()


    ##############################################
    # SIR model
    ##############################################

    # Total population, N.
    N = 18399000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 2997, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    #beta_proposed, gamma_proposed = 0.1, 0.19990000000000002
    beta_proposed, gamma_proposed = 0.15128413, 0.1
    # A grid of time points (in days)
    t = np.linspace(0, 300, 300)

    ##############################################
    # beta_proposed and gamma_proposed are found by manual trial and error
    ##############################################

    ##############################################
    # trying to find beta and gamma with some basic algorithm
    # and computing S, I, R with those found beta and gamma
    ##############################################

    beta_arr = np.arange(0.1, 0.2, 0.00001)
    gamma_arr = np.arange(0.01, 0.11, 0.0001)

    beta = 0
    gamma = 0

    min = 99999999999
    """
    for i in beta_arr:
        for j in gamma_arr:

            # The SIR model differential equations.
            def deriv(y, t, N, beta, gamma):
                S, I, R = y
                dSdt = -beta * S * I / N
                dIdt = beta * S * I / N - gamma * I
                dRdt = gamma * I
                return dSdt, dIdt, dRdt

            # Initial conditions vector
            y0 = S0, I0, R0
            # Integrate the SIR equations over the time grid, t.
            ret = odeint(deriv, y0, t, args=(N, i, j))
            S, I, R = ret.T

            I_cmp = I[0:len(days)]
            mse = I_cmp.mean(axis=None)
            if( mse < min):
                min = mse
                beta = i
                gamma = j

    ##############################################
    # computing S, I, R with beta_proposed, gamma_proposed
    ##############################################

    """
    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta_proposed, gamma_proposed))
    S, I, R = ret.T

    #print(beta)
    #print(gamma)            

    ##############################################
    # end of finding beta and gamma
    # from now on we only use S, I and R
    ##############################################

    alpha = 1
    I = (1/alpha)*I
    S = (1/alpha) * S
    R = (1/alpha) * R

    ##############################################
    # computing S_tilda, I_tilda and R_tilda
    ##############################################

    w = 1 # in interval[0,1] - some parameter in order to better represent the number of susceptible individuals
    S_tilda, I_tilda, R_tilda = tilda_functions(I, R, alpha, w, N, t)

    ##############################################
    # computing phi, delta, delta_bar and phi_bar matrices
    ##############################################

    phi = compute_phi(S, I, alpha, t)
    print("Shape for Phi matrix is: ", phi.shape)

    delta_v1 = compute_delta(phi, beta_proposed, gamma_proposed)
    print("Shape for Delta matrix is: ", delta_v1.shape)

    delta = compute_delta_v2(I_tilda, R_tilda, t)
    print("Shape for Delta matrix v2 is: ", delta.shape)

    
    theta = len(t) #length of time window
    ro = 0.999 #exponential decay weighting parameter
    
    delta_bar = compute_delta_bar(delta, theta, ro)
    print("Shape for Delta bar matrix is: ", delta_bar.shape)

    phi_bar = compute_phi_bar(phi, theta, ro)
    print("Shape for Phi bar matrix is: ", phi_bar.shape)

    ##############################################
    # calculating w, alpha, beta, gamma
    ##############################################

    w_star, alpha_star, beta_star, gamma_star = italianGS(I_tilda, R_tilda, alpha, ro, N, t)

    print("omega is: ", w_star)
    print("alpha is: ", alpha_star)
    print("beta is: ", beta_star)
    print("gamma is: ", gamma_star)

    print(I[:83])
    #print(I[83]/1000)



    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    #ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(days, cases_no/1000, alpha=0.5, lw=2, label='Real Infected')
    #ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,19000)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

