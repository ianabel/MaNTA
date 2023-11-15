# Component part of vtkConverter
import vtkmodules.all as vtk
import sys
import numpy as np

# Data should be a list of tuples ( "variableName", numpyArray with dims N_X,N_Y )
def write2DVTK( X, Y, filename, data ):
	vtkData = vtk.vtkUnstructuredGrid()
	grid = vtk.vtkPoints()
	nPoints = X.size * Y.size;
	vtkData.Allocate( nPoints )
	for ix in range(X.size):
		for iy in range(Y.size):
			x = X[ix]
			y = Y[iy]
			pointID = ix + iy*X.size
			grid.InsertPoint( pointID, x, y, 0.0 )
	vtkData.SetPoints(grid)
	for ix in range(X.size - 1):
		for iy in range(Y.size - 1):
			# Vertices in counter-clockwise order
			points = [ ix + iy*X.size, ix + 1 + iy*X.size, ix + 1 + (iy + 1)*X.size, ix + (iy + 1)*X.size ]
			vtkData.InsertNextCell( vtk.VTK_QUAD, 4, points )

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



def writeVTK( X, Y, filename, data, extrude = False ):
	if extrude:
		write3DVTK( X, Y, filename, data )
	else:
		write2DVTK( X, Y, filename, data )

