# %%

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def main():
    plt.rcParams.update({
        'font.family': 'serif',
    })
    electronMass = 9.1094e-31
    ionMass = 1.6726e-27
    protonMass = 1.6726e-27
    e_charge = 1.60217663e-19
    T0 = 1e3*e_charge
    n0 = 1e20
    p0= n0*T0

    data = Dataset('./MirrorPlasma.nc')#xr.open_dataset("./Config/LinearDiffusion.nc")
    Vars = data.groups

    
    # plt.figure()
    # plt.plot(data.groups["Var2"].variables["sigma"][5,:])
    # plt.show()
    t = np.array(data.variables["t"])
    x = data.variables["x"] 
    r = np.sqrt(np.array(x)/np.pi)
    #r = np.sqrt(np.array(x)*2)
    # for Var in Vars:
    #     plt.figure()
    #     for k in range(0,len(t),int(len(t)-1)):
    #         plt.plot(x[:],data.groups[Var].variables["sigma"][k,:])
    #     plt.title("sigma" + Var)
    #     plt.show()
    #     plt.figure()
    #     for k in range(0,len(t),int(len(t)-1)):
    #         plt.plot(x[:],data.groups[Var].variables["q"][k,:])
    #         plt.plot(x[:],data.groups[Var].variables["u"][k,:])
    #     plt.title("q" + Var)
    #     plt.show()

  

    plt.figure()
    ax = plt.axes()
    n = np.array(data.groups["Var0"].variables["u"])
    p_i = 2./3.*np.array(data.groups["Var2"].variables["u"])
    p_e  = 2./3.*np.array(data.groups["Var1"].variables["u"])
    omega = np.array(data.groups["Var3"].variables["u"])/(n*r*r)
    Voltage = np.array(data.variables["Voltage"])
    omega = omega[-1,:]
    Ti = np.divide(p_i,n)
    Te = np.divide(p_e,n)

    ax.plot(r,data.groups["Var0"].variables["u"][-1,:],label = r"$\hat{n}$")
   # ax.plot(r,data.groups["Var3"].variables["u"][-1,:],label = r"$\hat{h}$")
    ax.legend()
    plt.xlabel(r"$\hat{r}$")
    plt.figure()
    ax2 = plt.axes()
    ax2.plot(r,p_e[-1,:],label = r"$\hat{p}_e$")
    ax2.plot(r,p_i[-1,:],label = r"$\hat{p}_i$")
    ax2.plot(r,Te[-1,:],label = r"$\hat{T}_e$")
    ax2.plot(r,Ti[-1,:],label = r"$\hat{T}_i$")
    ax2.legend()
    plt.xlabel(r"$\hat{r}$")
    plt.figure()
    ax3 = plt.axes()
    ax3.plot(r,omega,label=r"$\omega$")
    ax3.legend()

    M0 = 26.0
    shape = 20.0
    Rmin = 0.2
    Rmax = 0.7
    omegaOffset = 5.0
    u_L = omegaOffset
    u_R = (omegaOffset * Rmin / Rmax)
    a = (np.arcsinh(u_L) - np.arcsinh(u_R)) / (Rmin - Rmax)
    b = (np.arcsinh(u_L) - Rmin / Rmax * np.arcsinh(u_R)) / (a * (Rmin / Rmax - 1))

    shape = 20.0
    C = 0.5 * (Rmin + Rmax);
    c = (np.pi / 2 - 3 * np.pi / 2) / (Rmin - Rmax)
    d = (np.pi / 2 - Rmin / Rmax * (3 * np.pi / 2)) / (c * (Rmin / Rmax - 1))
    coef = M0/C#(omegaOffset - M0 / C) * 1 / np.cos(c * (C - d))


    # h_i = np.array(data.groups["Var3"].variables["u"])
    # omega = h_i/(n*r*r)#np.sinh(a * (r - b)) - np.cos(c * (r - d)) * coef #* np.exp(-shape * (r - C) * (r - C))
    # omega = np.squeeze(omega[-1,:])
    #omega = np.sinh(a * (r - b)) - np.cos(c * (r - d)) * coef* np.exp(-shape * (r - C) * (r - C))
    w0 = np.sqrt(e_charge*1000/ionMass)
    M = r*omega/np.sqrt(Te)
    v = r*omega*w0
    plt.figure()
    ax3 = plt.axes()
    ax3.plot(r,v,label =r"$v_\theta$")
    ax3.legend()
    plt.xlabel(r"$\hat{r}$")
    plt.ylabel(r"$v_\theta (m/s)$")
    plt.figure()
    plt.plot(t,Voltage)
    
    plt.title("Voltage vs time")

    plt.figure()
    AlphaHeat = np.array(data.groups["Heating"].variables["AlphaHeating"])
    ViscousHeat = np.array(data.groups["Heating"].variables["ViscousHeating"])
    Pbrem = np.array(data.groups["Heating"].variables["RadiationLosses"])

    Parloss = np.array(data.groups["ParallelLosses"].variables["ParLoss"])
    Phi0 = np.array(data.groups["ParallelLosses"].variables["CentrifugalPotential"])
    plt.plot(r,AlphaHeat[-1,:])
    plt.title("Alpha heating")
    plt.figure()
    plt.plot(r,ViscousHeat[-1,:])
    plt.title("Viscous heating")
    plt.figure()
    plt.plot(r,Pbrem[-1,:])
    plt.title("Brem")
    plt.figure()
    plt.semilogy(r,Parloss[-1,:])
    plt.title("parallel losses")
    plt.figure()
    plt.plot(r,Phi0[-1,:])
    plt.title("phi0")

    data.close()

if __name__ == "__main__":
    main()
