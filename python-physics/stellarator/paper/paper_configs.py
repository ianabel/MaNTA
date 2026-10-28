fac = 1.0
st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": fac * 1.6e-2,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": fac * 3.0,
    "NBIWidth": 0.25,
    "ECHCenter": 0.0,
    "ECHPower": fac * 8.76e-2,
    "ECHWidth": 0.26,
    "EdgeTemperature": 0.2,
    "EdgeDensity": 0.2,
    "n0": 0.5,
    "T0": 1.0,
    "FusionFactor": 0.01,
    "evolveDensity": True,
    "useBatching": True,
}

st_config1 = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": fac * 0.02,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": fac * 5.5,
    "NBIWidth": 0.15,
    "ECHCenter": 0.0,
    "ECHPower": fac * 8.76e-2,
    "ECHWidth": 0.15,
    "EdgeTemperature": 0.3,
    "EdgeDensity": 0.2,
    "ADCoefficient": 0.0,
    "n0": 0.5,
    "T0": 1.0,
    "FusionFactor": 0.01,
    "evolveDensity": True,
    "useBatching": True,
}


fac = 1.0
high_col = {
    "ParticleSourceCenter": 0.5,
    "ParticleSourceHeight": fac * 1.2,
    "ParticleSourceWidth": 0.4,
    "NBICenter": 0.0,
    "NBIPower": fac * 300.0,
    "NBIWidth": 0.15,
    "ECHCenter": 0.0,
    "ECHPower": fac * 0.0,
    "ECHWidth": 0.1,
    "EdgeTemperature": 0.5,
    "EdgeDensity": 0.25,
    "ADCoefficient": 0.0,
    "n0": 1.5,
    "T0": 1.0,
    "FusionFactor": 1.0,
    "evolveDensity": True,
    # Er by current balance with a small diffusion rather than ambipolarity
    # node by node, so that one root is chosen where J(Er) = 0 has several.
    # Sized from an algebraic run of this case (stellarator_opteq4.dat): with
    # J' ~ n D_i / T_i from its fluxes, D_E * |curvature of Er| / J' stays
    # under 5% of Er out to rho ~ 0.98 only for D_E <~ 1e-6 -- Er's curvature
    # grows four decades towards the edge while J' falls -- and the zero-flux
    # edge layer, sqrt(D_E / J') ~ 3e-3, then sits inside the last cell.
    # ErInertia keeps tau's stabilisation (an added diffusivity ~ ErInertia *
    # tau * h) below D_E on the widest cell; see DiffusiveErFlux.
    "ErModel": "Diffusive",
    "ErDiffusion": 1e-6,
    "ErInertia": 5e-6,
    "useITGProxy": False,  # NN ITG heat flux (itg_wrapper.py); needs torch + ITG_MODEL_DIR
    "ITGScale": 1.0,  # gyro-Bohm is large in these units; scales the prediction
    "useBatching": True,
    "nchunks": 3,
}
