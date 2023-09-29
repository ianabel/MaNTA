# %%

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def main():

    data = Dataset('./Config/4VarCyl.nc')#xr.open_dataset("./Config/LinearDiffusion.nc")
    Vars = data.groups
    # plt.figure()
    # plt.plot(data.groups["Var2"].variables["sigma"][5,:])
    # plt.show()
    t = data.variables["t"]
    x = data.variables["x"] 
    
    for Var in Vars:
        plt.figure()
        for k in range(0,len(t),int(len(t)-1)):
            plt.plot(x[:],data.groups[Var].variables["sigma"][k,:])
        plt.title("sigma" + Var)
        plt.show()
        plt.figure()
        for k in range(0,len(t),int(len(t)-1)):
            plt.plot(x[:],data.groups[Var].variables["q"][k,:])
        plt.title("q" + Var)
        plt.show()

    

    plt.figure()
    ax = plt.axes()

    ax.plot(x[:],data.groups["Var0"].variables["u"][-1,:],label = "n")
    ax.plot(x[:],data.groups["Var1"].variables["u"][-1,:],label = "Pe")
    ax.plot(x[:],data.groups["Var2"].variables["u"][-1,:],label = "Pi")
    ax.plot(x[:],data.groups["Var3"].variables["u"][-1,:],label = "h")
    ax.legend()

    n = np.array(data.groups["Var0"].variables["u"])
    Gamma = -np.array(data.groups["Var0"].variables["sigma"])
    Vr = Gamma/n
    gradPe = np.array(data.groups["Var1"].variables["q"])
    gradPi = np.array(data.groups["Var2"].variables["q"])
    heating = gradPe*Vr
    heatingi = gradPi*Vr
    plt.figure()
    plt.plot(x[:],heating[-1,:])
    plt.plot(x[:],heatingi[-1,:])
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
