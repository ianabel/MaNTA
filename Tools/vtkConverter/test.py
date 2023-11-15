#!/usr/bin/env python3

import numpy as np
import vtkOutput as vo

X = np.array([0.0, 0.5, 1.0])
Y = np.array([0.0, 0.5, 1.0])

blob = np.array( [[0.0,0.0,0.0],[0.0,0.25,0.5],[0.0,0.5,1.0]] )

vo.write2DVTK( X, Y, "test.vtu", [("blob",blob)] )

