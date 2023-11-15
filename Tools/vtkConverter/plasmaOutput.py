#!/usr/bin/env python3

# Simple python script that takes MaNTA outputs and an equilibrium file and, if necessary, generates phi_0 at all
# points before writing a combined netCDF file

from netCDF4 import Dataset

import numpy as np
import scipy.interpolate
import scipy.optimize

class PlasmaClass:
    def __init__( self, ncFileManta, ncFileEq, tIndex = -1 ):
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
		  
		  # For the moment assume just a two species (4 variable) run
		  # and pull the last time index
		  self.PsiVals = np.array( MantaData.variables["Psi"] )
		  self.N_data = np.array( MantaData.group["n"].variable["u"][tIndex,:] )
		  self.P_data = []
		  self.P_data.insert( 0, MantaData.group["Pe"].variable["u"][tIndex,:] )
		  self.P_data.insert( 1, MantaData.group["Pi"].variable["u"][tIndex,:] )

		  self.m = [ ElectronMass, IonMass ]
		  self.Z = [ -1, 1 ]

		  self.Omega_Data = np.array( MantaData.group["Omega"].variable["u"][tIndex,:] )

		  self.omega = scipy.interpolate.Akima1DInterpolator( self.PsiVals, self.Omega_Data )
		  self.N_interpolant = [ scipy.interpolate.Akima1DInterpolator( self.PsiVals, self.N_data ), scipy.interpolate.Akima1DInterpolator( self.PsiVals, self.N_data ) ]
		  self.P_interpolant = [ scipy.interpolate.Akima1DInterpolator( self.PsiVals, self.P_data[0] ), scipy.interpolate.Akima1DInterpolator( self.PsiVals, self.P_data[1] ) ]



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
			Ndata[k] = self.N(k, psi)
			Tdata[k] = self.T(k, psi)
			rhoApprox[k] = self.Z[k] * Ndata[k] * np.exp( self.m[k] * omega**2 * Rval**2/ (2*Tdata[k]) )

		qn_func = (lambda phi,rho = rhoApprox,Z = self.Z,T = Tdata : np.sum( rhoApprox * np.exp( -Z * phi0 / T ) ) )
		phi0 = scipy.optimize.root_scalar( qn_func, method='secant', x0 = 0.0, x1 = 0.5 )
		return phi0
	
	def N( self, s, psi ):
		return self.N_interpolant[ s ]( psi )

	def T( self, s, psi ):
		return self.P_interpolant[ s ]( psi ) / self.N_interpolant[ s ]( psi )


