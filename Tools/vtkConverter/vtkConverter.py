#!/usr/bin/env python3
#

import numpy as np
import sys

# Usage ./vtkConverter.py <MaNTA netcdf file> <Equilibrium File> <Output VTK file>

if len(sys.argv) == 4:
    MantaFile = sys.argv[1]
    EqFile    = sys.argv[2]
    outFile   = sys.argv[3]
else:
    print("Usage ./vtkConverter.py <MaNTA netcdf file> <Equilibrium File> <Output VTK file>")
    sys.exit(1)


import plasmaOutput

# Plamsa.N and Plasma.T are functions taking (species index, psi)
# Plasma.Psi is a 2D array
# Plasma.R / Plasma.Z are 1D arrays 

Plasma = plasmaOutput.PlasmaClass( MantaFile, EqFile )

DensityData = np.zeros( (Plasma.R.size,Plasma.Z.size,Plasma.nSpecies) )
TemperatureData = np.zeros( (Plasma.R.size,Plasma.Z.size,Plasma.nSpecies) )
OmegaData = np.zeros( (Plasma.R.size,Plasma.Z.size) )

for i in range(Plasma.R.size):
	for j in range(Plasma.Z.size):
		psiVal = Plasma.Psi[i,j]
		phi0 = Plasma.calcPhi0(i,j)
		omega = Plasma.omega(psiVal)
		OmegaData[i,j] = omega
		for s in range(Plasma.nSpecies):
			T = Plasma.T(s,psiVal)
			TemperatureData[i,j,s] = T
			DensityData[i,j,s] = Plasma.N(s,psi) * np.exp( Plasma.m[s] * omega**2 * Plasma.R[i]**2 / (2.0 * T) - Plasma.Z[s] * phi0 / T )

data = [ ("Psi",OmegaData) ]

for s in range(Plasma.nSpecies):
    data.append( ("n"+Plasma.specie[s],DensityData[:,:,s] ) )
    data.append( ("T"+Plasma.specie[s],TemperatureData[:,:,s] ) )


import vtkOutput

vtkOutput.writeVTK(
