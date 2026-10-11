"""Helpers shared by the tests.

get_transport_system_as_module used to be a copy of the loader in manta/cli.py.
It delegates now, so the rule about where PythonModuleFile is resolved from has
one implementation rather than two that can drift.
"""

import os

from manta.cli import load_physics_modules

# The tests' inputs -- .conf files, .ref.nc references, case modules -- live
# here. The tests themselves run elsewhere; see conftest.py.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def data_path(name):
    return os.path.join(TESTS_DIR, name)


def get_transport_system_as_module(config_path):
    load_physics_modules(config_path)
    return config_path
