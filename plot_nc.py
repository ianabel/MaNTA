# %%

import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt

def main():

    data = xr.open_dataset("./Config/AutodiffTransportSystem.nc")
    x = data["x"]
    t = data["t"]

    plt.figure()
    plt.plot(x,data["Var0"][0,:])
 #   plt.contourf(x,t,data["Var0"])
    plt.show()
    data.close()

if __name__ == "__main__":
    main()
