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

    data = Dataset('./MirrorPlasmaDebug.nc')#xr.open_dataset("./Config/LinearDiffusion.nc")
    Vars = data.groups
    # plt.figure()
    # plt.plot(data.groups["Var2"].variables["sigma"][5,:])
    # plt.show()
    t = data.variables["t"]
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


    Rm = 3.3
    tau = p_i / p_e
    M2 = r * r * omega*omega * n/p_e

    phi0 =  1 / (1 + tau) * (1 / Rm - 1) * r*r*omega * omega / 2;

    Xe =  0.5 * (1 - 1 / Rm) * M2 * 1 / (tau + 1)
    Xi = 0.5 * tau / (1 + tau) * (1 - 1 / Rm) * M2
    # plt.figure()
    # ax4 = plt.axes()
    # ax4.plot(r,Xe[-1,:], label=r"$\Xi_e$")
    # ax4.legend()

    # plt.figure()
    
    # ax5 = plt.axes()
    # ax5.plot(r,omega, label=r"$\omega$")
    # ax5.plot(r,phi0[-1,:],label=r"$\phi_0$")
    # ax5.legend()

    n = np.squeeze(n[-1,:])*1e20
    TeV = np.squeeze(Te[-1,:]*1000)
    Pbrem = -2*1e6 * 1.69e-32 * (n * n) * 1e-12 * np.sqrt(TeV)
    sv = 3.68e-12 * np.power(TeV / 1000, -2. / 3.) * np.exp(-19.94 * np.power(TeV / 1000, -1. / 3.))
    Pfus = np.sqrt(1 - 1 / Rm)* 1e6*  5.6e-13 * n * n * 1e-12 * sv
    Ltot = 2
    Pbremtot = 2*np.pi*Ltot*np.trapz(r*Pbrem,r)
    Pfustot = 0.25*2*np.pi*Ltot*np.trapz(r*Pfus,r)

    Sigma = 2
    tau_hat = (1.0 / np.power(n/1e20, 5.0 / 2.0)) * (np.power(p_e, 3.0 / 2.0))
  


    Bmid = 4.5
    Om_e = e_charge*Bmid/electronMass
    lam = 23.0 - np.log(np.sqrt(n0 * 1e-6) / (T0/e_charge))
    taue0 = 3.44e11 * (1.0 / np.power(n0, 5.0 / 2.0)) * (np.power(p0 / e_charge, 3.0 / 2.0)) * 1 / lam
    taui0 = np.sqrt(2)*np.sqrt(ionMass/electronMass)*taue0
    Gamma0 = p0 / (electronMass * Om_e * Om_e * taue0)
    V0 = Gamma0 / n0
    L = 1
    n= n/1e20
    Spast = (-2 * n * Sigma / np.sqrt(np.pi) * 1 / tau_hat * 1 / np.log(Sigma * Rm) * np.exp(-Xe) / Xe)
    Spasttot = 2*np.pi*Ltot*np.trapz(r*np.squeeze(n0*V0/L*L/(taue0*V0)*Spast[-1,:]),r)
    Ppaste = p0*V0/L*L/(taue0*V0)*Te*(1+Xe)*Spast
    Ppasti = p0*V0/L*Ti*(1+Xi)*Spast
    Ppastetot = 2*np.pi*Ltot*np.trapz(r*np.squeeze(Ppaste[-1,:]),r)

    Ppasti = p0*V0/L*L/(taui0*V0)*Ti*(1+Xi)*Spast
    Ppastitot = 2*np.pi*Ltot*np.trapz(r*np.squeeze(Ppasti[-1,:]),r)
    V = np.sqrt(T0/ionMass)*Bmid*np.trapz(r,omega*r*r)
    Te = np.squeeze(Te[-1,:])
    rm = r[np.squeeze(Te)==np.max(Te)]
    M = np.squeeze(M[-1,:])
    Mmid = np.squeeze(M[r==rm])
    Tiev = Ti[-1,:]*T0/e_charge
    Teev = Te*T0/e_charge
    Temid = np.squeeze(Teev[r==rm])
    Timid = np.squeeze(Tiev[r==rm])



    Vp = 2*np.pi
    c2 = r*r*Vp 
    
    domegadR = (a*np.cosh(a*(r-b)) + coef*(c*np.sin(c*(r-d))*np.exp(-shape * (r - C) * (r - C))+np.cos(c*(r-d))*2*shape*(r-C)*np.exp(-shape * (r - C) * (r - C))))
    domegadV = c2*domegadR/(Vp*r)
    Pvis = np.sqrt(ionMass/electronMass)*1/np.sqrt(2)*1/tau_hat[-1,:]* 3. / 10. * p_i[-1,:]*domegadV*domegadV

    Pvis = np.squeeze(Pvis)

    Pvistot = 2*np.pi*Ltot*np.trapz(r*Pvis,r)
    
    print("Total bremsstrahlung power: ",Pbremtot)
    print("Total alpha power: ",Pfustot)
    print("Total parallel particle losses", Spasttot)
    print("Total parallel power losses: ",Ppastetot+Ppastitot)
    print("Total potential drop: ",V)
    print("Central Mach number: ", Mmid)
    print("Total viscous heating: ", Pvistot)
    print("Central ion temp", Timid)
    print("Central electron temp", Temid)

    sourceStrength = 25.0
    sourceCenter =  2.0
    sourceWidth = 0.6
    Sn = n0*V0/L*sourceStrength * np.exp(-1 / sourceWidth * (np.array(x) - sourceCenter) * (np.array(x) - sourceCenter))
    plt.figure()
    plt.plot(r,Sn)

    Sntot = 2*np.pi*Ltot*np.trapz(r*Sn,r)
    print("Particle Source",Sntot)
    plt.xlabel(r"$\hat{r}$")
    plt.ylabel(r"$S_n (m^{-3}s^{-1})$")
    
    plt.figure()
    plt.plot(r,Pvis)
    plt.xlabel(r"$\hat{r}$")
    plt.ylabel(r"$P_{vis}$")
    #ax6.plot(r,Spast[-1,:])

    # h_i = np.array(data.groups["Var3"].variables["u"])
    # w = np.sqrt(np.divide(h_i,r*r*n))
    # ax.plot(r,r*w[-1,:],label=r"$\hat{R}\hat{\omega}_i$")
    # ax.plot(r,data.groups["Var3"].variables["u"][-1,:],label = r"$\hat{h}_i$")
   

    
   
    
    # Gamma = np.array(data.groups["Var0"].variables["sigma"])
    # Vr = Gamma/n
    # gradPe = np.array(data.groups["Var1"].variables["q"])
    # gradPi = np.array(data.groups["Var2"].variables["q"])
    # heating = gradPe*Vr
    # heatingi = gradPi*Vr
    # plt.figure()
    # plt.plot(x[:],Vr[-1,:])
    # plt.plot(x[:],Vr[-1,:])
    # plt.show()

   
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


    # plt.figure()
    # #plt.plot(x,data["Var0"][0,:])
    # plt.contourf(x[:],t[:],data.groups["Var1"].variables["u"][:,:])
    # plt.show()
    
#     # plt.figure()
#     # for i in range( int(length/10)):
    
#     #     plt.plot(x,data["Var0"][4*i,:])

#     # plt.show()

    data.close()

if __name__ == "__main__":
    main()
