# %%
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np


# %% [markdown]
# Load file

# %%
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10
})
electronMass = 9.1094e-31
ionMass = 2*1.6726e-27
protonMass = 1.6726e-27
e_charge = 1.60217663e-19
T0 = 1e3*e_charge

data = Dataset('../runs/CMFX1.nc')#xr.open_dataset("./Config/LinearDiffusion.nc")

t = np.array(data.variables["t"])
x = np.array(data.variables["x"] )


# %% [markdown]
# Load data

# %%
Tau = np.array(data.variables["Tau"])

ionMass = 1.6726e-27
mu0 = 4*np.pi*10**-7

taunorm = np.array(data.variables["Tau"])
print(taunorm)
Lnorm = np.array(data.variables["Lnorm"])
T0 = np.array(data.variables["T0"])
n0 = np.array(data.variables["n0"])
# Ldev = np.array(data.variables["L_z"])
L_fl = np.array(data.variables["L"])
Ldev = L_fl[0,:]

r = np.array(data.variables["R"])
r = r[0,:]

p0 = n0*T0
Ir = -np.array(data.variables["Current"])[-1]

Current = -np.array(data.variables["Current"])
ComputedCurrent = np.array(data.variables["ComputedCurrent"])

omega0 = np.sqrt(T0/ionMass)

B = np.array(data.variables["B"])
Rm = np.array(data.variables["Rm"])


n = np.array(data.groups["Var0"].variables["u"])
dn = np.array(data.groups["Var0"].variables["q"])
Gamma = np.array(data.groups["Var0"].variables["sigma"])
p_i = 2./3.*np.array(data.groups["Var1"].variables["u"])
p_e  = 2./3.*np.array(data.groups["Var2"].variables["u"])
dp_i = 2./3.*np.array(data.groups["Var1"].variables["q"])
dp_e  = 2./3.*np.array(data.groups["Var2"].variables["q"])
qi = np.array(data.groups["Var1"].variables["sigma"])
qe = np.array(data.groups["Var2"].variables["sigma"])
Pi = np.array(data.groups["Var3"].variables["sigma"])
L = np.array(data.groups["Var3"].variables["u"])


phi = np.array(data.variables["ElectrostaticPotential"])
sr = np.array(data.groups["DimensionlessNumbers"].variables["ShearingRate"])
rhoTi = np.array(data.groups["DimensionlessNumbers"].variables["RhoTi"])
rhoN = np.array(data.groups["DimensionlessNumbers"].variables["RhoN"])
rhoTe = np.array(data.groups["DimensionlessNumbers"].variables["RhoTe"])
rhoL = np.array(data.groups["DimensionlessNumbers"].variables["RhoL"])
eta_e = np.array(data.groups["DimensionlessNumbers"].variables["eta_e"])
col = np.array(data.groups["DimensionlessNumbers"].variables["Collisionality"])

nAD  = np.array(data.groups["ArtificialDiffusion"].variables["DensityArtificialDiffusion"])
piAD  = np.array(data.groups["ArtificialDiffusion"].variables["IonPressureArtificialDiffusion"])
peAD  = np.array(data.groups["ArtificialDiffusion"].variables["ElectronPressureArtificialDiffusion"])
LAD  = np.array(data.groups["ArtificialDiffusion"].variables["AngularMomentumArtificialDiffusion"])

LTi = np.array(data.groups["GradientScaleLengths"].variables["LTi"])

omega = L/(n*r*r)
    
pfluxderiv = np.abs((dp_i)/p_e + 3./2.*dn/n)-np.abs(0.5*dp_e/p_e)

Voltage = np.array(data.variables["Voltage"])

Ti = p_i/n
Te = p_e/n

phi1 = Ti*np.array(data.variables["AuxVariable0"])


tau = Ti/Te
M = omega*r/np.sqrt(Te)


# %% [markdown]
# Load more data

# %%
Bz = 0.35
Q_i =np.array(data.groups["Var1"].variables["sigma"])
Q_e =np.array(data.groups["Var2"].variables["sigma"])
Gamma =np.array(data.groups["Var0"].variables["sigma"])
Pi = np.array(data.groups["Var3"].variables["sigma"])
omega = L/(n*r*r)
# Vprime = 2*np.pi*Ldev/Bz

Neutrals = data.groups["Neutrals"]

R_CX = np.array(Neutrals.variables["ChargeExchange"])
R_ion = np.array(Neutrals.variables["Ionization"])
plt.figure()
plt.plot(r,R_ion[-1,:]*taunorm/n0)

CX_Heat = Ti[-1,:]*T0*R_CX[-1,:]
CX_Momentum = T0*(r*omega[-1,:])**2*R_CX[-1,:]

# Pin = #2*np.pi*Ldev*np.trapezoid(jr/Vprime*omega[-1,:]*omega0*r*Bz,r)    
HeatSources = data.groups["Heating"]


Prad = np.array(HeatSources.variables["RadiationLosses"])

plt.figure()
plt.plot(r,Prad[-1,:])


Prad = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*Prad[-1,:],r)

Palpha = np.array(HeatSources.variables["AlphaHeating"])

plt.figure()
plt.plot(r,Palpha[-1,:])

Palpha = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*Palpha[-1,:],r)

Pcyc = np.array(HeatSources.variables["CyclotronLosses"])
plt.figure()
plt.plot(r,Pcyc[-1,:])

Pcyc = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*Pcyc[-1,:],r)

Pvis = np.array(HeatSources.variables["ViscousHeating"])
# Pvis[Pvis > 6e4] = 6e4
Pvis = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*Pvis[-1,:],r)
PCX = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*CX_Heat,r)
LCX = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*CX_Momentum,r)


PpotI = np.array(HeatSources.variables["IonPotentialHeating"])
PpotI = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*PpotI[-1,:],r)

PpotE = np.array(HeatSources.variables["ElectronPotentialHeating"])
PpotE = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*PpotE[-1,:],r)


ParallelLosses = data.groups["ParallelLosses"]
Pepar = np.array(ParallelLosses.variables["ElectronParLoss"])
Pepar = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*Pepar[-1,:],r)

Pipar = np.array(ParallelLosses.variables["IonParLoss"])
Pipar = 2*np.pi*np.trapezoid(r*Ldev*Lnorm*p0/(taunorm)*Pipar[-1,:],r)

Lpar = np.array(ParallelLosses.variables["AngularMomentumLosses"])


data.close()

# %% [markdown]
# Plot base outputs

# %%
fig, axs = plt.subplots(1,3)

ax = axs[0]
ax.plot(r,n[-1,:],label=r"$\hat{n}$")
# ax.legend()
ax.set_title(r"n ($10^{20}$  $m^{-3}$)")
# ax.set_xlabel(r"$\Hat{V}$")
ax.set_xlabel(r"$R(m)$")
ax.set(adjustable='box')
ax = axs[1]
ax.plot(r,r*omega0/1000*omega[-1,:],label = r"$\hat{\omega}$")
# ax.legend()
ax.set_title(fr"$v_\theta$($km/s$)")
# ax.set_xlabel(r"$\Hat{V}$")\
ax.set_xlabel(r"$R(m)$")
ax.set(adjustable='box')
ax = axs[2]
ax.plot(r,Ti[-1,:],label = r"$T_i$",color="tab:blue")
ax.plot(r,Te[-1,:],label = r"$T_e$",color="tab:red")
ax.legend()
ax.set_title(fr"T ($keV$)")
# ax.set_xlabel(r"$\Hat{V}$")
ax.set_xlabel(r"$R(m)$")
ax.set(adjustable='box')
fig.set_size_inches(5,5/2.5)
fig.tight_layout()
plt.savefig("cmfx.svg")

# %%
fig, axs = plt.subplots(1,3)

ax = axs[0]
ax.plot(r,n[-1,:],label=r"$\hat{n}$")
ax.legend()
ax.set_title(r"Density ($10^{20}$  $1/m^3$)")
# ax.set_xlabel(r"$\Hat{V}$")
ax.set_xlabel(r"$r(m)$")
ax.set(adjustable='box')
ax = axs[1]
ax.plot(r,L[-1,:],label = r"$\hat{n}\hat{\omega}R^2$")
ax.legend()
ax.set_title(fr"Angular momentum density (${round(ionMass*n0*omega0,2)}$  $kg/m\cdot s$)")
# ax.set_xlabel(r"$\Hat{V}$")
ax.set_xlabel(r"$r(m)$")
ax.set(adjustable='box')
ax = axs[2]
ax.plot(r,p_i[-1,:],label = r"$\hat{p_i}$")
ax.plot(r,p_e[-1,:],label = r"$\hat{p_e}$")
ax.legend()
ax.set_title(fr"Ion and electron pressure (${round(p0/1000,2)}$  $kPa$)")
# ax.set_xlabel(r"$\Hat{V}$")
ax.set_xlabel(r"$r(m)$")
ax.set(adjustable='box')
fig.set_size_inches(18,5)
# plt.savefig("cmfx.eps")

# %% [markdown]
# 

# %% [markdown]
# Plot derived outputs

# %%
fig, axs = plt.subplots(2,2)
ax = axs[0,0]
# ax.plot(r,n[0,:],label="t=0")
ax.plot(r,n[-1,:],label = r"$\hat{n}$")
ax.set_box_aspect(1)   
# sol = np.array(data.groups["MMS"].variables["Var0"]);
# ax.plot(r,sol[-1,:],label="MMS solution")    
ax.legend()
ax.set_title(r"Density ($10^{20}$  $1/m^3$)")
ax.set_xlabel(r"r $(m)$")



# plt.figure()
ax = axs[0,1]
ax.plot(r,Te[-1,:],label = r"$\hat{T}_e$")
ax.plot(r,Ti[-1,:],label = r"$\hat{T}_i$")
ax.set_box_aspect(1)
ax.set_xlabel(r"r $(m)$")
ax.legend()
ax.set_title(r"Ion and Electron Temperatures ($keV$)")


w0 = np.sqrt(e_charge*1000/ionMass)
M = r*omega/np.sqrt(Te)
v = r*omega*w0
vthe = np.sqrt(Te[-1,:]*T0/electronMass)
SRe = r/vthe*np.gradient(v[-1,:])/(r[1]-r[0])
# plt.figure()
ax = axs[1,0]
ax.plot(r,v[-1,:]/1000,label =r"$v_\theta$")
ax.set_box_aspect(1)
ax.legend()
ax.set_xlabel(r"r $(m)$")
# ax.set_ylabel(r"$v_\theta (m/s)$")
ax.set_title(r"Azumuthal velocity ($km/s$)")

ax = axs[1,1]
ax.plot(r,M[-1,:],label=r"M")
ax.legend()
ax.set_xlabel(r"r $(m)$")
ax.set_title("Mach Number")
height = 10
fig.set_size_inches(height, height)


# %% [markdown]
# Plot voltage

# %% [markdown]
# 

# %%
plt.figure()
plt.plot(x,L[-1,:]/n[-1,:])


# %%
plt.figure()
plt.plot(t*Tau,Voltage)

plt.title("Voltage vs time")
plt.figure()
plt.plot(r,B[0,:])
plt.title("B")

plt.figure()
plt.plot(r,L_fl[0,:])
plt.title("L")

plt.figure()
plt.plot(r,Rm[0,:])
plt.title("Mirror ratio")

fig,ax = plt.subplots()

ax.plot(t,Current,color="r",label = "Current")
ax.plot(t,ComputedCurrent,color="b",label="Computed current")
ax.legend()

plt.figure()
plt.plot(t*Tau,Voltage/ComputedCurrent)
plt.title("Resistance")




# %% [markdown]
# Plot dimensionless numbers

# %%
plt.figure()
ax = plt.axes()
ax.semilogy(x,np.abs((sr[-1,:])/LTi[-1,:]))
# ax.set_ylim([0,10])
ax.set_xlabel(r"$\hat{V}$")
# ax.axhline(-1.0,color='r')
ax.axhline(1.0,color='r')
# plt.plot(r,SRe)
plt.title(r"$(c_s/\nabla v_\theta)/L_{T,i}$")
plt.savefig("shear_diff.eps",dpi=300)

plt.figure()
ax = plt.axes()
ax.semilogy(x,np.abs(sr[-1,:]))
# ax.set_ylim([0,10])
ax.set_xlabel(r"$\hat{V}$")
# ax.axhline(-1.0,color='r')
ax.axhline(1.0,color='r')
# plt.plot(r,SRe)
plt.title(r"$(c_s/\nabla v_\theta)$")


plt.figure()
ax = plt.axes()
ax.semilogy(x,np.abs(rhoN[-1,:]))
ax.set_xlabel(r"$\hat{V}$")
# ax.set_ylim([0,1.0])

ax.axhline(1.0,color='r')
# ax.set_yscale("log")
# plt.plot(r,SRe)
plt.title(r"$\rho_i/L_N$")

plt.figure()
ax = plt.axes()
ax.semilogy(x,np.abs(rhoTi[-1,:]))
ax.set_xlabel(r"$\hat{V}$")
# ax.set_ylim([0,1.0])

ax.axhline(1.0,color='r')
# ax.set_yscale("log")
# plt.plot(r,SRe)
plt.title(r"$\rho_i/L_Ti$")

plt.figure()
ax = plt.axes()
ax.semilogy(x,np.abs(rhoTe[-1,:]))
ax.set_xlabel(r"$\hat{V}$")
# ax.set_ylim([0,1.0])

ax.axhline(1.0,color='r')
# ax.set_yscale("log")
# plt.plot(r,SRe)
plt.title(r"$\rho_i/L_Te$")
# plt.savefig("rholn_diff.eps",dpi=300)

plt.figure()
ax = plt.axes()
ax.plot(x,col[-1,:])
ax.set_xlabel(r"$\hat{V}$")
# plt.plot(r,SRe)
plt.title(r"$\nu_{ii}L_\parallel/c_s$")

plt.figure()
ax = plt.axes()
ax.semilogy(x,np.abs(eta_e[-1,:]))
ax.set_xlabel(r"$\hat{V}$")
ax.axhline(1.0,color='r')
# ax.set_ylim([0,1.0])
# plt.plot(r,SRe)
plt.title(r"$eta_e$")

# %%
fig, axs = plt.subplots(2,2)
ax = axs[0,0]
ax.plot(r,Gamma[-1,:],label=r"$\Gamma$ w smoothing")
ax.plot(r,Gamma[-1,:]+nAD[-1,:],label=r"$\Gamma$ w/o smoothing")
ax.legend()

threshold = 0.2

rshade = r[np.abs(rhoN[-1,:])>=threshold]
if (len(rshade>0)):
    h = r[1]-r[0]
    hd = np.diff(rshade)
    i = np.where(hd>h)[0]
    i+=1

    rshade = np.split(rshade,i)
    for arr in rshade:
        ax.axvspan(arr[0],arr[-1],alpha = 0.25,color="red")

ax = axs[0,1]
ax.plot(r,qi[-1,:],label=r"$q_i$ w smoothing")
ax.plot(r,qi[-1,:]+piAD[-1,:],label=r"$q_i$ w/o smoothing")

rshade = r[np.abs(rhoTi[-1,:])>=threshold]
if (len(rshade>0)):
    h = r[1]-r[0]
    hd = np.diff(rshade)
    i = np.where(hd>h)[0]
    i+=1

    rshade = np.split(rshade,i)
    for arr in rshade:
        ax.axvspan(arr[0],arr[-1],alpha = 0.25,color="red")

ax.legend()

ax = axs[1,0]
ax.plot(r,qe[-1,:],label=r"$q_e$ w smoothing")
ax.plot(r,qe[-1,:]+peAD[-1,:],label=r"$q_e$ w/o smoothing")

rshade = r[np.abs(eta_e[-1,:])>=0.8]

if (len(rshade>0)):
    h = r[1]-r[0]
    hd = np.diff(rshade)
    i = np.where(hd>h)[0]
    i+=1

    rshade = np.split(rshade,i)
    for arr in rshade:
        ax.axvspan(arr[0],arr[-1],alpha = 0.25,color="red")

ax.legend()

ax = axs[1,1]
ax.plot(r,Pi[-1,:],label=r"$\pi_i$ w smoothing")
ax.plot(r,Pi[-1,:]+LAD[-1,:],label=r"$\pi_i$ w/o smoothing")

rshade = r[np.abs(rhoL[-1,:])>=threshold]

if (len(rshade>0)):
    h = r[1]-r[0]
    hd = np.diff(rshade)
    i = np.where(hd>h)[0]
    i+=1

    rshade = np.split(rshade,i)
    for arr in rshade:
        ax.axvspan(arr[0],arr[-1],alpha = 0.25,color="red")

ax.legend()

height = 10
fig.set_size_inches(height, height)

plt.title("Fluxes showing added artificial diffusion")

# %% [markdown]
# Compute averaged quantites

# %%

# Voltage = np.trapezoid(omega[-1,:]*np.sqrt(T0/ionMass)/Vprime,x)
   
# print("other voltage: ",Voltage2)

# Pi -= omega*r*r*Gamma
B = B[0,:]
AlfvenSpeed = B/np.sqrt(ionMass*n*n0*mu0)

Pin = Voltage[-1]*Ir

beta = p0*(p_i+p_e)/(B**2/(2*mu0))

Ti = p_i[-1,:]/n[-1,:]
Te = p_e[-1,:]/n[-1,:]
M = omega[-1,:]*r/np.sqrt(Te)
Ma = omega0*omega*r/AlfvenSpeed

rmid = 0.5*(r[0]+r[-1])

# jRadial =  jr*rmid/Bz

Tiavg = np.mean(Ti)
Teavg = np.mean(Te)
Mavg = np.mean(M)
omegaavg = np.sqrt(T0/ionMass)*np.mean(omega[-1,:])
navg = np.mean(n[-1,:])

# normalizing quantities

print("Normalizing time (s): ", taunorm)
print("Normalizing pressure (Pa): ", p0)

# Variables
print("Average Density (10^20 m^-3): ", navg)
print("Average Electron temperature (keV): ", Teavg)
print("Peak electron temperature (keV): ", np.max(Te))

print("Average ion Temperature (keV): ", Tiavg)
print("Peak ion temperature (keV): ", np.max(Ti))

print("Average Mach number: ", Mavg)
print("Average Alfven Mach number: ",np.mean(Ma[-1,:]))
print("Average rotational frequency (10^6 1/s): ", omegaavg/1e6)
print("Max rotational frequency (10^6 1/s): ", omega0*np.max(omega[-1,:])/1e6)
print("Voltage (V): ",Voltage[-1])
print("Radial current (A): ", Ir)
print("beta: ",100*np.mean(beta[-1,:]))

print("\n")

# Sources
PLpar = 2*np.pi*np.trapezoid(Ldev*Lnorm*r*p0/(taunorm)*omega[-1,:]*Lpar[-1,:],r)
Ptorque = 2*np.pi*np.trapezoid(Ldev*Lnorm*r*p0/(taunorm)*omega[-1,:]*np.gradient(Pi[-1,:],x[1]-x[0]),r)

peqnorm = (p0/taunorm)
Ptorque_boundary = peqnorm*(omega[-1,-1]*Pi[-1,-1]-omega[-1,0]*Pi[-1,0])
# Piomega = Lnorm*T0/taunorm*omega[-1,:]*Pi[-1,:]
# Ptorque = 2*np.pi*Ldev*(Piomega[-1] - Piomega[0]) 


PclasI = np.abs(peqnorm*(Q_i[-1,0]-Q_i[-1,-1]))
PclasE = np.abs(peqnorm*(Q_e[-1,0]-Q_e[-1,-1]))

IonHeatLoss = Pipar + PclasI + PCX
ElectronHeatLoss = Pepar+PclasE+Prad+Pcyc

StoredIonEnergy = 3./2.*2*np.pi*np.trapezoid(Ldev*Lnorm*r*p_i[-1,:]*p0,r)
StoredElectronEnergy = 3./2.*2*np.pi*np.trapezoid(Ldev*Lnorm*r*p_e[-1,:]*p0,r)

# TotalEnergyFlow = Ptorque_boundary + PclasI + PclasE - 

print("Input power (W): ", Pin)
print("Parallel power losses (W): ",PLpar)

# print("Classical power loss (W): ", peqnorm*(np.abs(omega[-1,0]*Pi[-1,0])+np.abs(omega[-1,1]*Pi[-1,-1])))
print("Power losses from viscous torque (W): ", Ptorque)
print("Power lost at the boundaries (W): ", Ptorque_boundary)
print("Power losses from charge exchange (W): ", LCX)

print("Pin - Pout (W): ",Pin - (Ptorque+PLpar+LCX))

print("Viscous torque - viscous heating (W)", Ptorque - Pvis)

print("Total bremsstrahlung loss (W): ", Prad)
print("Total cyclotron loss (W): ", Pcyc)
print("Total viscous heating (W): ", Pvis)

print("Total alpha heating (W): ", Palpha)


print("Total ion parallel heat loss (W): ", Pipar)
print("Total electron parallel heat loss (W): ", Pepar)

print("Total ion potential heating (W): ", PpotI)
print("Total electron potential heating (W): ", PpotE)

print("Classical ion heat loss (W): ", PclasI)
print("Classical electron heat loss (W): ", PclasE)

print("Charge exchange heat loss", PCX)

print("Stored electron energy (J): ", StoredElectronEnergy)
print("Stored Ion Energy (J): ", StoredIonEnergy)

print("Qin - Qout: ",Pvis+Palpha-np.abs(Pipar+Pepar+PclasI+PclasE+Prad+PCX+Pcyc-PpotI- PpotE))
print("Energy confinement time (ms): ",1000* (StoredElectronEnergy+StoredIonEnergy)/(IonHeatLoss+ElectronHeatLoss))
# print("Particle Confinment time (ms): ", np.mean(n[-1.:]*n0)/)



