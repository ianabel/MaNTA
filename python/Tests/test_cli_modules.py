"""The config's physics-module keys, and how they resolve.

Two forms are supported. `PythonModule` names an importable module and is the
documented one. `PythonModuleFile` names a file, which is what every config in
this repository was written for, and is resolved *relative to the config file*
-- not to the current directory, which is what the retired PyManta script did
and why those configs only ever worked from one place.
"""

import sys

import pytest

from manta.cli import load_physics_modules


CASE_SOURCE = """
import manta

LOADED = True
REGISTERED = False

def registerTransportSystems():
    global REGISTERED
    REGISTERED = True
"""


def _write_case(directory, module_filename="case.py", **conf):
    (directory / module_filename).write_text(CASE_SOURCE)
    keys = "\n".join(f'{k} = "{v}"' for k, v in conf.items())
    config = directory / "run.conf"
    config.write_text(f'[configuration]\nTransportSystem = "X"\n{keys}\n')
    return config


def test_module_file_resolves_relative_to_the_config_not_the_cwd(tmp_path):
    config = _write_case(tmp_path, PythonModuleName="casemod",
                         PythonModuleFile="case.py")

    # conftest.py runs every test with cwd = python/Tests, so a cwd-relative
    # reading of "case.py" cannot possibly succeed here. That is the point.
    load_physics_modules(config)

    assert sys.modules["casemod"].LOADED


def test_the_legacy_registration_hook_is_called(tmp_path):
    config = _write_case(tmp_path, PythonModuleName="hookmod",
                         PythonModuleFile="case.py")

    load_physics_modules(config)

    assert sys.modules["hookmod"].REGISTERED, (
        "every example module in the tree registers through "
        "registerTransportSystems() rather than at import"
    )


def test_the_module_name_defaults_to_the_file_stem(tmp_path):
    config = _write_case(tmp_path, module_filename="stemcase.py",
                         PythonModuleFile="stemcase.py")

    load_physics_modules(config)

    assert sys.modules["stemcase"].LOADED


def test_a_missing_file_names_the_key_that_is_wrong(tmp_path):
    config = tmp_path / "run.conf"
    config.write_text('[configuration]\nPythonModuleFile = "absent.py"\n')

    with pytest.raises(SystemExit) as excinfo:
        load_physics_modules(config)

    assert "PythonModuleFile" in str(excinfo.value)


def test_a_broken_module_does_not_stay_in_sys_modules(tmp_path):
    """A half-executed module left behind would make the next import of that
    name succeed silently against a broken object."""
    (tmp_path / "boom.py").write_text("raise ValueError('deliberate')\n")
    config = tmp_path / "run.conf"
    config.write_text(
        '[configuration]\nPythonModuleName = "boom"\nPythonModuleFile = "boom.py"\n'
    )

    with pytest.raises(ValueError):
        load_physics_modules(config)

    assert "boom" not in sys.modules


def test_a_dotted_module_still_works(tmp_path):
    config = tmp_path / "run.conf"
    config.write_text('[configuration]\nPythonModule = "json"\n')

    load_physics_modules(config)  # no exception


def test_extra_modules_are_imported_too(tmp_path):
    config = tmp_path / "run.conf"
    config.write_text("[configuration]\n")

    load_physics_modules(config, extra_modules=["json"])  # no exception
