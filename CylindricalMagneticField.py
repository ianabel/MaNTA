import numpy as np
import toml
from netCDF4 import Dataset

class CylindricalMagneticField():
    def __init__(self,B0,Rm0,nPoints,config) -> None:
        self.B0 = B0
        self.Rm0 = Rm0
        f = toml.load(config)

        self.Ri = np.sqrt(f["configuration"]["Lower_boundary"]/np.pi)
        self.Ro = np.sqrt(f["configuration"]["Upper_boundary"]/np.pi)

        self.m = -self.B0/(2*(self.Ro-self.Ri))

        self.R = np.linspace(self.Ri,self.Ro,nPoints)
        
    # Linear B - decreases from a value of B0 at Ri to a value of 0.5*B0 at Ro
    def Bz(self) -> np.ndarray:
        return self.B0 + self.m*(self.R - self.Ri)
    
    def Psi(self) -> np.ndarray:

        return self.B0*self.R**2/2.+ self.m*(self.R**3/3. - self.Ri*self.R**2/2.)
    
    def Rm(self) -> np.ndarray:
        return self.Bz()/self.B0*self.Rm0
    
    def getR(self) -> np.ndarray:
        return self.R

def main():
    nPoints = 300
    B0 = 3.0
    Rm0 = 3.0
    config = "./Config/MirrorPlasmaDebug.conf"
    B = CylindricalMagneticField(B0,Rm0,nPoints,config)

    ncfile = Dataset("./PhysicsCases/Bfield.nc",mode="w",format="NETCDF4_CLASSIC")
    
    ncfile.createDimension('R',nPoints)
    R = ncfile.createVariable('R',np.float64,('R',))
    Bz = ncfile.createVariable('Bz',np.float64,('R',))
    Rm = ncfile.createVariable('Rm',np.float64,('R',))
    Psi = ncfile.createVariable('Psi',np.float64,('R',))

    R.units = 'm'
    R[:] = B.getR()

    Bz.units = 'T'
    Bz[:] = B.Bz()

    Rm[:] = B.Rm()

    Psi[:] = B.Psi()

    print(ncfile)
    ncfile.close()

if __name__ == "__main__":
    main()


    