"""Dump what a MaNTA port of test-w7x-relu-ke needs, computed by T3D itself."""
import sys, numpy as np
from t3d.trinity_lib import TrinityEngine
from t3d.geometry_models.VMEC import VmecReader

inp, out = sys.argv[1], sys.argv[2]
eng = TrinityEngine(inp)
sp, geo, norms, grid = eng.species, eng.geometry, eng.norms, eng.grid
vm = VmecReader(geo.vmec_wout)
H, e = sp["hydrogen"], sp["electron"]
print("ref species:", sp.ref_species.type, "n_evolve:", sp.n_evolve_keys, "T_evolve:", sp.T_evolve_keys)
print("masses H, e:", H.mass, e.mass, "Z:", H.Z, e.Z)
print("Btor", geo.Btor, "a_minor", geo.a_minor, "a_vmec", geo.a_minor_vmec, "t_ref", norms.t_ref, "P_ref_MWm3", norms.P_ref_MWm3)
print("relu H:", H.n_relu_critical_gradient0, H.n_relu_critical_gradient1, H.n_relu_flux_slope, H.p_relu_critical_gradient0, H.p_relu_critical_gradient1, H.p_relu_flux_slope)
print("relu e:", e.n_relu_critical_gradient0, e.n_relu_critical_gradient1, e.n_relu_flux_slope, e.p_relu_critical_gradient0, e.p_relu_critical_gradient1, e.p_relu_flux_slope)
print("Sp aux e: width", e.Sp_aux_width, "center", e.Sp_aux_center, "int", e.Sp_aux_int)
m = list(eng.flux_models.get_models_list())[0]
print("model D_neo", m.D_neo, "power", m.p, "noise", m.noise)
np.savez(out,
         vmec_rho=np.asarray(vm.rho), vmec_area=np.asarray(vm.area), vmec_grad_rho=np.asarray(vm.grad_rho),
         vmec_dVdrho=np.asarray(vm.dVdrho), a_vmec=geo.a_minor_vmec, a_minor=geo.a_minor, Btor=geo.Btor,
         t_ref=norms.t_ref, P_ref_MWm3=norms.P_ref_MWm3, m_H=H.mass, m_e=e.mass, Z_H=H.Z, Z_e=e.Z,
         rho=grid.rho, n_e=e.n().profile, p_H=H.p().profile, p_e=e.p().profile,
         Sp_aux_e=e.Sp_labeled["aux"].profile, area_grid=geo.area_grid.profile, grho_grid=geo.grho_grid.profile,
         rho_edge=grid.rho_edge, D_neo=m.D_neo, power=m.p,
         relu_H=[H.n_relu_critical_gradient0, H.n_relu_flux_slope, H.p_relu_critical_gradient0, H.p_relu_flux_slope],
         relu_e=[e.n_relu_critical_gradient0, e.n_relu_flux_slope, e.p_relu_critical_gradient0, e.p_relu_flux_slope],
         Sp_aux=[e.Sp_aux_width, e.Sp_aux_center, e.Sp_aux_int])
