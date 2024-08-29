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
        for Var in Vars: 

            if (Var.startswith("Var")):
                plt.figure()
                ax = plt.axes()
                y = np.array(data.groups[Var].variables["u"])
                ax.plot(x,y[-1,:],label=Var)
                if (include_initial):
                    ax.plot(x,y[0,:],label=Var+", t = 0")
                ax.legend()
                plt.title("u")
        if (plot_grid):
            for x in Grid:
                plt.axvline(x,label="_grid",color="red",linestyle="--",alpha=0.25)
       
    
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

        for Var in Vars:
            if (Var.startswith("Var")):
                plt.figure()
                ax = plt.axes()
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
    x = np.array(data.variables["x"])

    if ("MMSSource" in data.groups):
        MMSSource = data.groups["MMSSource"]
        for var in MMSSource.variables:
            plt.figure()
            s = np.array(MMSSource.variables[var])
            plt.plot(x,s[-1,:])
            plt.xlabel("x")
            plt.ylabel("MMS source")
            plt.title("MMS source for " + var)


    MMS = data.groups["MMS"]
    plt.figure()
    ax = plt.axes()
    for var in MMS.variables:
        s = np.array(MMS.variables[var])
        sac = np.array(data.groups[var].variables["u"])
        diff = np.amax(np.abs(s - sac),axis=1)
        ax.semilogy(t,diff,label = var + " error")
        start = round(0.1*diff.size)
        fit = np.polyfit(t[start:],np.log(diff[start:]),1)
        ax.semilogy(t,np.exp(fit[1]+fit[0]*t),label=var + " fit")
        print("Error growth rate = " + "{:.3f}".format(fit[0]))

    ax.legend()
    plt.xlabel("t")
    plt.ylabel("error")

    data.close()

def plot_diagnostics(fname):
    data = Dataset(fname)
    t = np.array(data.variables["t"])
    x = np.array(data.variables["x"])
    for group in data.groups:
        if (not group.startswith("Var") and not group.startswith("MMS") and not group.startswith("Grid")):
            for var in data.groups[group].variables:
                y = np.array(data.groups[group].variables[var])
                plt.figure()
                ax = plt.axes()
                ax.plot(x,y[-1,:],label = var)
                ax.plot(x,y[0,:],label = var + " t=0")
                ax.legend()
                plt.title(data.groups[group].description)
                plt.xlabel("x")



    data.close()


def main():
    fname = "./MirrorPlasmaTestRERUN.nc"
    plot_nc(fname,False,True,include_initial=True)
    fname = "./MirrorPlasmaTest.nc"
    plot_nc(fname,False,True,include_initial=True)
    # plot_MMS(fname)
    #plot_diagnostics(fname)
    plt.show()
    

if __name__ == "__main__":
    main()

# %%
