#!/usr/bin/env python

import numpy as np
import toml
from netCDF4 import Dataset

class CylindricalMagneticField():
    def __init__(self,B0,Rm0,nPoints,config) -> None:
        self.B0 = B0
        self.Rm0 = Rm0
        f = toml.load(config)

        self.Vi = np.sqrt(f["configuration"]["Lower_boundary"])
        self.Vo = np.sqrt(f["configuration"]["Upper_boundary"])
        self.L_z = f["MirrorPlasma"]["Lz"]
        self.V = np.linspace(self.Vi,self.Vo,nPoints)

        self.Ro = np.sqrt(self.Vo/(np.pi*self.L_z))
        self.Ri = np.sqrt(self.Vi/(np.pi*self.L_z))

        self.m = 0.0*self.B0/(2*(self.Ro-self.Ri))

        
    # Linear B - decreases from a value of B0 at Ri to a value of 0.5*B0 at Ro
    def R(self) -> np.ndarray:
        return np.sqrt(self.V/(np.pi*self.L_z))
    
    def dRdV(self) -> np.ndarray:
        return 1.0 / (2 * np.pi * self.L_z * self.R())

    def Bz(self) -> np.ndarray:
        return self.B0 + self.m*(self.R() - self.Ri)
    
    def Psi(self) -> np.ndarray:
        R = self.R()
        return self.B0*R**2/2.+ self.m*(R**3/3. - self.Ri*R**2/2.)
    
    def VPrime(self) ->np.ndarray:
        return 2 * np.pi * self.L_z / self.Bz()
    
    def Rm(self) -> np.ndarray:
        return self.Bz()/self.B0*self.Rm0
    

def main():
    nPoints = 300
    B0 = 4.5
    Rm0 = 3.3
    config = "./Config/CMFX.conf"
    B = CylindricalMagneticField(B0,Rm0,nPoints,config)

    ncfile = Dataset("./Bfield.nc",mode="w",format="NETCDF4")
    
    ncfile.createDimension('V',nPoints)
    V_var = ncfile.createVariable('V',np.float64,('V',))
    Vprime =ncfile.createVariable('VPrime',np.float64,('V',))
    L_z = ncfile.createVariable('L',np.float64,('V',))
    R = ncfile.createVariable('R',np.float64,('V',))
    dRdV = ncfile.createVariable('dRdV',np.float64,('V',))
    Bz = ncfile.createVariable('Bz',np.float64,('V',))
    Rm = ncfile.createVariable('Rm',np.float64,('V',))
    Psi = ncfile.createVariable('Psi',np.float64,('V',))

    R.units = 'm'
    V_var[:] = B.V
    R[:] = B.R()
    dRdV[:] = B.dRdV()

    Bz.units = 'T'
    Bz[:] = B.Bz()

    Vprime[:] = B.VPrime()

    Rm[:] = B.Rm()

    Psi[:] = B.Psi()

    L_z[:] = B.L_z*np.ones(B.V.shape)

    print(ncfile)
    ncfile.close()

if __name__ == "__main__":
    main()


    
