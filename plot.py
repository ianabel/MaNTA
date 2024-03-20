# %%

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def plot_nc(fname,plot_u = True, plot_q = False, plot_sigma = False, include_initial = False):

    data = Dataset(fname)
    Vars = data.groups
    t = np.array(data.variables["t"])
    x = np.array(data.variables["x"] )
    if (plot_u):
        plt.figure()
        ax = plt.axes()
        for Var in Vars: 
            try:
                y = np.array(data.groups[Var].variables["u"])
                ax.plot(x,y[-1,:],label=Var)
                if (include_initial):
                    ax.plot(x,y[0,:],label=Var+", t = 0")
            except:
                continue
        ax.legend()
    if (plot_q):
        plt.figure()
        ax = plt.axes()
        for Var in Vars: 
            try:
                y = np.array(data.groups[Var].variables["q"])
                ax.plot(x,y[-1,:],label=Var)
                if (include_initial):
                    ax.plot(x,y[0,:],label=Var+", t = 0")
            except:
                continue
        ax.legend()
    if (plot_sigma):
        plt.figure()
        ax = plt.axes()
        for Var in Vars: 
            try:
                y = np.array(data.groups[Var].variables["sigma"])
                ax.plot(x,y[-1,:],label=Var)
                if (include_initial):
                    ax.plot(x,y[0,:],label=Var+", t = 0")
            except:
                continue
        ax.legend()



def main():
    fname = "./LinearDiffusion.nc"
    plot_nc(fname)

if __name__ == "__main__":
    main()
