# %%

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def plot_nc(fname,plot_u = True, plot_q = False, plot_sigma = False, plot_grid= False, include_initial = False):

    data = Dataset(fname)
    Vars = data.groups
    Grid = np.array(data.groups["Grid"].variables["CellBoundaries"])
    t = np.array(data.variables["t"])
    x = np.array(data.variables["x"] )
    if (plot_u):
        plt.figure()
        ax = plt.axes()
        for Var in Vars: 
            if (Var.startswith("Var")):
                y = np.array(data.groups[Var].variables["u"])
                ax.plot(x,y[-1,:],label=Var)
                if (include_initial):
                    ax.plot(x,y[0,:],label=Var+", t = 0")
    
        ax.legend()
        if (plot_grid):
            for x in Grid:
                plt.axvline(x,label="_grid",color="red",linestyle="--",alpha=0.25)
       
        plt.title("u")
    if (plot_q):
        plt.figure()
        ax = plt.axes()
        for Var in Vars: 
            if (Var.startswith("Var")):
                y = np.array(data.groups[Var].variables["q"])
                ax.plot(x,y[-1,:],label=Var)
                if (include_initial):
                    ax.plot(x,y[0,:],label=Var+", t = 0")

        ax.legend()
        plt.title("q")
    if (plot_sigma):
        plt.figure()
        ax = plt.axes()
        for Var in Vars: 
            if (Var.startswith("Var")):
                y = np.array(data.groups[Var].variables["sigma"])
                ax.plot(x,y[-1,:],label=Var)
                if (include_initial):
                    ax.plot(x,y[0,:],label=Var+", t = 0")

        ax.legend()
        plt.title("sigma")

    data.close()

def plot_MMS(fname):
    data = Dataset(fname)
    t = np.array(data.variables["t"])

    MMS = data.groups["MMS"]
    plt.figure()
    ax = plt.axes()
    for var in MMS.variables:
        s = np.array(MMS.variables[var])
        sac = np.array(data.groups[var].variables["u"])
        diff = np.amax(np.abs(s - sac),axis=1)
        ax.semilogy(t,diff,label=var)


def main():
    fname = "./LinearDiffusion.nc"
    plot_nc(fname,True,True,True,False,True)
    plot_MMS(fname)

if __name__ == "__main__":
    main()

# %%
