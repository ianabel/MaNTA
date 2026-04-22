import MaNTA

from Stellarator import StellaratorTransport
from yancc_wrapper import yancc_data

# %%
st_config = {
    "SourceCenter": 0.0,
    "SourceHeight": 150.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.1,
    "EdgeDensity": 0.0,
    "n0": 0.5,
}
# runner = MaNTA.Runner(st)

# # %%
solver_config = {
    "OutputFilename": "stellarator",
    "Polynomial_degree": 3,
    "Grid_size": 6,
    "tau": 1.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 0.9,
    "Relative_tolerance": 0.01,
    "delta_t": 0.01,
    "restart": False,
    "solveAdjoint": False, 
}

config = {
    "Stellarator": st_config,
    "Solver": solver_config,
}

points =  MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])
Density = lambda x : (st_config["n0"] - st_config["EdgeDensity"]) * (1 - x*x) + st_config["EdgeDensity"]

yancc_wrapper = yancc_data.from_eq(points, Density=Density)

st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)

# runner.configure(config)

# %%
tFinal = 1.0
import time
import datetime

start = time.time()
finalState = st.run()

end = time.time()
time_duration = datetime.timedelta(seconds=end-start)
print("Elapsed time:")
print(time_duration)





# %%
G, G_p = st.runAdjointSolve()
print("Stored energy: " + str(G))
print(G_p)

# %%


# %%



