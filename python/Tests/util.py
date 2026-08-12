"""Loading a config's physics module, for the tests that drive runManta.

This used to be a copy of the loader in manta/cli.py. It delegates now, so the
rule about where PythonModuleFile is resolved from has one implementation
rather than two that can drift.
"""

from manta.cli import load_physics_modules


def get_transport_system_as_module(config_path):
    load_physics_modules(config_path)
    return config_path
