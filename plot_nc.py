# %%

import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def main():

    data = Dataset('./Config/LinearDiffusion.nc')#xr.open_dataset("./Config/LinearDiffusion.nc")
    Vars = data.groups
    for Var in Vars:
        
        plt.figure()
        plt.plot(data.groups[Var].variables["q"][0,:])
        plt.show()

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


#     plt.figure()
#     #plt.plot(x,data["Var0"][0,:])
#     plt.contourf(x,t,data["Var1"])
#     plt.show()
    
#     # plt.figure()
#     # for i in range( int(length/10)):
    
#     #     plt.plot(x,data["Var0"][4*i,:])

#     # plt.show()

    data.close()

if __name__ == "__main__":
    main()
