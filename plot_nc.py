# %%

import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt

def main():

    data = xr.open_dataset("./Config/4VarCyl.nc")
    x = data["x"]
    t = data["t"]
    length = t.shape[0]

    plt.figure()
    ax = plt.axes()
    ax.plot(x,data["Var1"][-1,:],label="electron pressure")
    ax.plot(x,data["Var2"][-1,:],label="Ion pressure")
    ax.legend()
    plt.show()


    # plt.figure()
    # #plt.plot(x,data["Var0"][0,:])
    # plt.contourf(x,t,data["Var0"])
    # plt.show()
    
    # plt.figure()
    # for i in range( int(length/10)):
    
    #     plt.plot(x,data["Var0"][4*i,:])

    # plt.show()

    data.close()

if __name__ == "__main__":
    main()
