import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def load_set(path):

    cols=["date","cases","hospitalised", "deaths"]
    df = pd.read_csv(path, sep=",", header=1, names=cols)

    return df

if __name__ == "__main__":

    
    path = Path("D:\\Dev\\repos\\covid19-ml\\sets\\nyc\\case-hosp-death.csv")
    set = load_set(path)

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

    plt.figure()
    plt.plot(days, cases_no)
    #plt.show()

    # Total population, N.
    N = 200000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 9, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta, gamma = 0.38, 0.1
    # A grid of time points (in days)
    t = np.linspace(0, 100, 100)

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
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,220)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
