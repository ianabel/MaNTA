#!/usr/bin/env python3

# Simple python script that takes MaNTA outputs and an equilibrium file and, if necessary, generates phi_0 at all
# points before writing a combined netCDF file

from netCDF4 import Dataset

import numpy as np
import scipy.interpolate
import scipy.optimize

class PlasmaClass:
    def __init__( self, ncFileManta, ncFileEq ):
        MantaData = Dataset(ncFileManta, "r", format="NETCDF4")
        EqData = Dataset(ncFileEq, "r", format="NETCDF4")
		  self.R = np.array( EqData.variables["R"] )
		  self.Z = np.array( EqData.variables["Z"] )
		  self.Psi = np.array( EqData.variables["Psi"] )
		  self.B = np.array( EqData.variables["B"] )
		  if "phi0" in EqData.variables:
		    self.phi0 = np.array( EqData.variables["phi0"] )
		  else:
		    self.phi0 = None
		  EqData.close()
		  

	def calcPhi0( self, i, j ):
		# If we read in phi0, assume it was kosher
		if self.phi0 is not None:
		   return self.phi0[i,j]

		Rval = self.R[i]
		psi = self.Psi[i,j]
		omega = self.omega(psi)
		Ndata = np.zeros( (self.nSpecies) )
		Tdata = np.zeros( (self.nSpecies) )
		rhoApprox = np.zeros( (self.nSpecies) )

		for k in range(Ndata.size):
			Ndata[k] = self.N[k](psi)
			Tdata[k] = self.T[k](psi)
			rhoApprox[k] = self.Z[k] * Ndata[k] * np.exp( self.m[k] * omega**2 * Rval**2/ (2*Tdata[k]) )

		qn_func = (lambda phi,rho = rhoApprox,Z = self.Z,T = Tdata : np.sum( rhoApprox * np.exp( -Z * phi0 / T ) ) )
		phi0 = scipy.optimize.root_scalar( qn_func, method='secant', x0 = 0.0, x1 = 0.5 )
		return phi0



# Assume Plamsa.N[i] and Plasma.T[i] are scipy interpolants
# Plasma.Psi is a 2D array
# Plasma.R / Plasma.Z are 1D arrays 

Plasma = PlasmaClass( MantaFile, EqFile )

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
			T = Plasma.T[s](psiVal)
			TemperatureData[i,j,s] = T
			DensityData[i,j,s] = Plasma.N[s] * np.exp( Plasma.m[s] * omega**2 * Plasma.R[i]**2 / (2.0 * T) - Plasma.Z[s] * phi0 / T )

outFile = Dataset(outputFileName, "w", format="NETCDF4")

