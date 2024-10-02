# Component part of vtkConverter
import vtkmodules.all as vtk
import sys
import numpy as np
from netCDF4 import Dataset

# Data should be a list of tuples ( "variableName", numpyArray with dim N_X )
def write1DVTK( X, filename, data ):
	vtkData = vtk.vtkUnstructuredGrid()
	grid = vtk.vtkPoints()
	nPoints = X.size;
	vtkData.Allocate( nPoints )
	for ix in range(X.size):
        x = X[ix]
        pointID = ix
        grid.InsertPoint( pointID, x, 0.0, 0.0 )
	vtkData.SetPoints(grid)
	for ix in range(X.size - 1):
        # Vertices in counter-clockwise order
        points = [ ix, ix + 1 ]
        vtkData.InsertNextCell( vtk.VTK_LINE, 2, points )

	print("Number of fields ",len(data))
	for jData in range(len(data)):
		array = vtk.vtkDoubleArray()
		array.SetNumberOfComponents(1)
		array.SetNumberOfTuples(nPoints)
		if type(data[jData]) is not tuple:
			print("Error")
			sys.exit(1)
		if len(data[jData]) != 2:
			print("2-tuple required")
			sys.exit(1)
		array.SetName(data[jData][0])
		dataNumpyArray = data[jData][1]

		for jPoint in range(nPoints):
			dataTuple = dataNumpyArray.flatten()[jPoint]
			print(dataTuple)
			array.SetTuple(jPoint, [dataTuple])

		vtkData.GetPointData().AddArray(array)
	
	writer = vtk.vtkXMLUnstructuredGridWriter()
	writer.SetFileName(filename)
	writer.SetInputData(vtkData)
	writer.Write()

MantaFile = sys.argv[1]

MantaData = Dataset(MantaFile,"r")

nVars = int(MantaData.variables["nVariables"])

xData = MantaData.variables["x"]

write1DVTK( xData, "blat.vtk", ( "u", np.asarray( MantaData.groups["Var0"].variables["u"][-1,:] ) ) )


