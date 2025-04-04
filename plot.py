# %%

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def plot_nc(fname,plot_u = True, plot_q = False, plot_sigma = False, plot_aux = False,plot_scalars = False, plot_grid= False, include_initial = False):
  
    data = Dataset(fname)
    print(data)
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
                    for cell in Grid:
                        ax.axvline(cell,label="_grid",color="red",linestyle="--",alpha=0.25)
       
    
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

    if (plot_aux):

        plt.figure()
        ax = plt.axes()
        y = np.array(data.variables["AuxVariable0"])
        ax.plot(x,y[-1,:],label="aux")
        if (include_initial):
            ax.plot(x,y[0,:],label="aux"+", t = 0")

        ax.legend()
        plt.title("aux")

    if (plot_scalars):
        Vars = data.variables
        for Var in Vars:
            if (Var.startswith("Scalar")):
                plt.figure()
                ax = plt.axes()
                y = np.array(data.variables[Var])
                #y2 = np.array(data.variables["Voltage"])
                ax.plot(t,y,"ro")
                #ax.plot(t,y2)
                plt.title(Var)

        

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
    # print(data)
    # sig = np.array(data.groups["Var0"].variables["sigma"])
    # phi0 = np.array(data.variables["dPhi0dV"])
    # plt.figure()
    # ax = plt.axes()
    # ax.plot(x,phi0[-1,:],label ="dphi0")
    # ax.plot(x,phi0[0,:],label = "dphi0" + " t=0")
    # ax.legend()
    # plt.title("dphi0dv")
    # plt.xlabel("x")
    # phi1 = np.array(data.variables["dPhi1dV"])
    # plt.figure()
    # ax = plt.axes()
    # ax.plot(x,phi1[-1,:],label ="dphi1")
    # ax.plot(x,phi1[0,:],label = "dphi1" + " t=0")
    # ax.legend()
    # plt.title("dphi1dv")
    # plt.xlabel("x")
    # plt.figure()
    # plt.plot(x,sig[-1,:]*(-phi0[-1,:]+phi1[-1,:]))
    for group in data.groups:
        # if (not group.startswith("Var") and not group.startswith("MMS") and not group.startswith("Grid") and not group.startswith("Scalar")):
        #     for var in data.groups[group].variables:
        #         y = np.array(data.groups[group].variables[var])
        #         plt.figure()
        #         ax = plt.axes()
        #         ax.plot(x,y[-1,:],label = var)
        #         ax.plot(x,y[0,:],label = var + " t=0")
        #         ax.legend()
        #         plt.title(data.groups[group].description)
        #         plt.xlabel("x")

        if (group.startswith("Scalar")):
            for var in data.groups[group].variables:
                plt.figure()
                ax = plt.axes()
                y = np.array(data.groups[group].variables[var])
                ax.plot(t,y,label=var)
                ax.legend()


 

    data.close()


def main():
    fname = "./runs/CMFX3.nc"
    plot_nc(fname,plot_u=True,plot_grid=True,plot_scalars=True,include_initial=False)
    # fname = "./MirrorPlasmaTest.nc"
    #plot_nc(fname,False,False,include_initial=True)
    # plot_MMS(fname)
    # plot_diagnostics(fname)
    plt.show()
    

if __name__ == "__main__":
    main()

# %%
