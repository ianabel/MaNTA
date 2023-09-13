# %%

import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt

def main():

    data = xr.open_dataset("./Config/3VarCyl.nc")
    x = data["x"]
    t = data["t"]
    length = t.shape[0]

    plt.figure()
    #plt.plot(x,data["Var0"][0,:])
    plt.contourf(x,t,data["Var0"])
    plt.show()
    
    plt.figure()
    for i in range( int(length/10)):
    
        plt.plot(x,data["Var0"][4*i,:])

    plt.show()

    data.close()

if __name__ == "__main__":
    main()
