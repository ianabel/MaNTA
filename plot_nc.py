# %%

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def main():

    data = Dataset('./Config/3VarMirror.nc')#xr.open_dataset("./Config/LinearDiffusion.nc")
    Vars = data.groups
    # plt.figure()
    # plt.plot(data.groups["Var2"].variables["sigma"][5,:])
    # plt.show()
    t = data.variables["t"]
    x = data.variables["x"] 
    r = np.sqrt(np.array(x)/np.pi)
    #r = np.sqrt(np.array(x)*2)
    for Var in Vars:
        plt.figure()
        for k in range(0,len(t),int(len(t)-1)):
            plt.plot(x[:],data.groups[Var].variables["sigma"][k,:])
        plt.title("sigma" + Var)
        plt.show()
        plt.figure()
        for k in range(0,len(t),int(len(t)-1)):
            plt.plot(x[:],data.groups[Var].variables["q"][k,:])
            plt.plot(x[:],data.groups[Var].variables["u"][k,:])
        plt.title("q" + Var)
        plt.show()

    

    plt.figure()
    ax = plt.axes()
    n = np.array(data.groups["Var0"].variables["u"])
    p_i = np.array(data.groups["Var2"].variables["u"])
    p_e  = np.array(data.groups["Var1"].variables["u"])
    Ti = np.divide(p_i,n)
    Te = np.divide(p_e,n)

    ax.plot(r,data.groups["Var0"].variables["u"][-1,:],label = r"$\hat{n}$")
    ax.plot(r,data.groups["Var1"].variables["u"][-1,:],label = r"$\hat{p}_e$")
    ax.plot(r,data.groups["Var2"].variables["u"][-1,:],label = r"$\hat{p}_i$")
    ax.plot(r,Ti[-1,:],label = r"$\hat{T}_i$")
    M0 = 23.5
    shape = 10.0
    Rmin = 0.5
    Rmax = 1.0
    omegaOffset = 3.7
    C = 0.5 * (Rmin + Rmax);
    c = (np.pi / 2 - 3 * np.pi / 2) / (Rmin - Rmax)
    d = (np.pi / 2 - Rmin / Rmax * (3 * np.pi / 2)) / (c * (Rmin / Rmax - 1))
    coef = (omegaOffset - M0 / C) * 1 / np.cos(c * (C - d))
    omega =  omegaOffset - np.cos(c * (r - d)) * coef * np.exp(-shape * (r - C) * (r - C))

    M = r*omega/np.sqrt(Te)
    ax.plot(r,r*omega,label =r"$r\omega$")

    # h_i = np.array(data.groups["Var3"].variables["u"])
    # w = np.sqrt(np.divide(h_i,r*r*n))
    # ax.plot(r,r*w[-1,:],label=r"$\hat{R}\hat{\omega}_i$")
    # ax.plot(r,data.groups["Var3"].variables["u"][-1,:],label = r"$\hat{h}_i$")
   

    
    ax.legend()
    plt.xlabel(r"$\hat{r}$")
    
    Gamma = np.array(data.groups["Var0"].variables["sigma"])
    Vr = Gamma/n
    gradPe = np.array(data.groups["Var1"].variables["q"])
    gradPi = np.array(data.groups["Var2"].variables["q"])
    heating = gradPe*Vr
    heatingi = gradPi*Vr
    plt.figure()
    plt.plot(x[:],Vr[-1,:])
    plt.plot(x[:],Vr[-1,:])
    plt.show()

   
    # plt.show()
    # for Var in Vars:
    #     plt.figure()
    #     plt.plot(data.groups[Var].variables["u"][0,:])
    #     plt.plot(data.groups[Var].variables["u"][-1,:])
    #     plt.show()
        
    #     plt.figure()
    #     plt.plot(data.groups[Var].variables["q"][0,:])
    #     plt.plot(data.groups[Var].variables["q"][-1,:])
    #     plt.show()

    #     plt.figure()
    #     plt.plot(data.groups[Var].variables["sigma"][0,:])
    #     plt.plot(data.groups[Var].variables["sigma"][-1,:])
    #     plt.show()



#     plt.figure()
#     ax = plt.axes()
#     ax.plot(x,data["Var0"][0,:],label="density")
#     ax.plot(x,data["Var1"][0,:],label="electron pressure")
#     ax.plot(x,data["Var2"][0,:],label="Ion pressure")
#    # ax.plot(x,data["Var3"][0,:],label="momentum")
#     ax.legend()
#     plt.title("t = 0")
#     plt.show()

#     plt.figure()
#     ax = plt.axes()
#     ax.plot(x,data["Var0"][-1,:],label="density")
#     ax.plot(x,data["Var1"][-1,:],label="electron pressure")
#     ax.plot(x,data["Var2"][-1,:],label="Ion pressure")
#   #  ax.plot(x,data["Var3"][-1,:],label="momentum")
#     ax.legend()
#     plt.title("t = 1")
#     plt.show()


    plt.figure()
    #plt.plot(x,data["Var0"][0,:])
    plt.contourf(x[:],t[:],data.groups["Var1"].variables["u"][:,:])
    plt.show()
    
#     # plt.figure()
#     # for i in range( int(length/10)):
    
#     #     plt.plot(x,data["Var0"][4*i,:])

#     # plt.show()

    data.close()

if __name__ == "__main__":
    main()
