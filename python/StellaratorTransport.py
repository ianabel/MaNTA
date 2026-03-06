# %%

import MaNTA

from Stellarator import StellaratorTransport

# %%
st_config = {
    "SourceCenter": 0.05,
    "SourceHeight": 40.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.3,
    "EdgeDensity": 0.1,
    "n0": 0.5,
}

st = StellaratorTransport(st_config)
runner = MaNTA.Runner(st)

# %%
config = {
    "OutputFilename": "stellarator2",
    "Polynomial_degree": 4,
    "Grid_size": 4,
    "tau": 1.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 0.9,
    "Relative_tolerance": 0.01,
    "delta_t": 0.1,
    "restart": False,
}

runner.configure(config)

# %%

tFinal = 1.0
import time
import datetime

start = time.time()
runner.run(tFinal)

end = time.time()
time_duration = datetime.timedelta(seconds=end-start)
print("Elapsed time:")
print(time_duration)

# %%



