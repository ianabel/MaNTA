import MaNTA

from Stellarator import StellaratorTransport

# %%
st_config = {
    "SourceCenter": 0.0,
    "SourceHeight": 50.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.5,
    "EdgeDensity": 0.1,
    "n0": 0.5,
}

st = StellaratorTransport(st_config)
runner = MaNTA.Runner(st)

# %%
config = {
    "OutputFilename": "stellarator",
    "Polynomial_degree": 3,
    "Grid_size": 6,
    "tau": 1.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 0.9,
    "Relative_tolerance": 0.01,
    "delta_t": 0.05,
    "restart": False,
    "solveAdjoint": True, 
}

runner.configure(config)

# %%
tFinal = 1.0
import time
import datetime

start = time.time()
finalState = runner.run_ss()

end = time.time()
time_duration = datetime.timedelta(seconds=end-start)
print("Elapsed time:")
print(time_duration)


# %%
import matplotlib.pyplot as plt

plt.plot(finalState["Variable"])


# %%
G, G_p = runner.runAdjointSolve()
print("Stored energy: " + str(G))
print(G_p)

# %%


# %%



