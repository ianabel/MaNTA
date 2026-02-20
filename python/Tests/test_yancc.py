import pytest


import sys
sys.path.append("../")
import MaNTA

from Stellarator import StellaratorTransport

import os
os.environ.pop("LD_LIBRARY_PATH", None)

config = {"SourceCenter": 0.5,
          "SourceHeight": 1.0,
          "SourceWidth": 1.0,
          "EdgeTemperature": 0.5,
          "EdgeDensity": 0.1,
          "n0": 1.0}

grid = MaNTA.Grid()

testPoint = 0.5
Tnorm = 1.0e3
nNorm = 1e20
temperature = 1.5e3 # in eV
dTdVn = -1.5e3 # derivative wrt normalized volume 
Vnorm = 0.3 # radial coordinate
state = {"Variable": [temperature/Tnorm],
         "Derivative": [dTdVn/Tnorm]}

t = 0.0
index = 0

def test_yancc():
    st = StellaratorTransport(config, grid)

    print(st.SigmaFn(index, state, Vnorm,t))

    print(st.dSigmaFn_du(index, state, Vnorm, t))

    print(st.dSigmaFn_dq(index, state, Vnorm, t))

test_yancc()

    




