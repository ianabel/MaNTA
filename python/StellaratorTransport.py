import MaNTA

from Stellarator import StellaratorTransport

# %%
st_config = {
    "SourceCenter": 0.0,
    "SourceHeight": 30.0,
    "SourceWidth": 0.2,
    "EdgeTemperature":0.5,
    "EdgeDensity": 0.1,
    "n0": 0.5,
}

st = StellaratorTransport(st_config)

# runner = MaNTA.Runner(st)

# # %%
# config = {
#     "OutputFilename": "stellarator",
#     "Polynomial_degree": 3,
#     "Grid_size": 6,
#     "tau": 1.0, 
#     "Lower_boundary": 0.0,
#     "Upper_boundary": 1.0,
#     "Relative_tolerance": 0.01,
#     "delta_t": 0.001,
#     "restart": True,
#     "solveAdjoint": True, 
# }

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



