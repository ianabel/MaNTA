# %%

import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt

def main():

    data = xr.open_dataset("MatrixDiffusion.nc")
    x = np.linspace(-1,1,301)
    print(data)
    # data = Dataset("LinearDiffusion.nc","r",format="NETCDF4")
    # t = data.dimensions["t"]
    # x = data.dimensions["x"]
    # nVars = 1
    # Var = data.variables["Var0"]
    # t = data["t"]
    # print(data)
    # y = Var[-1,:]
    plt.figure()
    plt.plot(x,data["Var0"][0,:])
    plt.show()
    data.close()

if __name__ == "__main__":
    main()
