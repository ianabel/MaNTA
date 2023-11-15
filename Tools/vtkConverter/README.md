# Description

vtkConverter is a python-based tool to convert the output of a MaNTA simulation to a 2D or 3D plot of denstiy, temperature, angular velocity.
This can either work with MaNTA output that contains magnetic field information, or can use a separate netcdf file containing the flux function and the magnetic field, projected onto a grid.

## Prerequisites

To use this script you will need

- Python netcdf bindings
- Python vtk bindings

