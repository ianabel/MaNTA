"""The `manta` command.

The standalone C++ binary can only run physics cases that were linked into it.
This entry point exists so that a case written in Python, living in the user's
own package, can be run the same way:

    manta myrun.conf

with the config naming the module that defines and registers it:

    [configuration]
    TransportSystem = "MyCase"
    PythonModule = "mypackage.mycase"

The module is imported for its side effects -- it is expected to call
`manta.registerPhysicsCase(name, cls)` at import, exactly as a C++ case
registers itself during static initialisation -- and control then passes to the
same `runManta` the binary uses.

A config written for the older tooling names the module by *file* instead, and
that still works. The path is resolved beside the config file:

    [configuration]
    TransportSystem  = "MyCase"
    PythonModuleName = "mycase"
    PythonModuleFile = "mycase.py"

In either form, a module that defines `registerTransportSystems()` has it called
after import. That is the convention every example in this repository uses;
registering at import is the documented one.
"""

import argparse
import importlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import manta


def _configuration_of(config_path):
    """The config's [configuration] table, or an empty one."""
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        # Let runManta report it, so the message is the same either way.
        return {}
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"manta: {config_path} is not valid TOML: {e}") from e

    return config.get("configuration", {})


def _load_module_from_file(config_path, conf):
    """Import the module named by PythonModuleFile, by path.

    The path is resolved relative to the config file rather than to the current
    directory. Every config in this repository is written as though that were
    already true; the retired PyManta script resolved against the cwd, which is
    why they only ever ran from one directory.
    """
    relative = conf["PythonModuleFile"]
    path = Path(config_path).parent / relative
    name = conf.get("PythonModuleName") or path.stem

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(
            f"manta: PythonModuleFile {relative!r} (looked for {path}) is not "
            "a loadable Python module"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        del sys.modules[name]
        raise SystemExit(
            f"manta: PythonModuleFile {relative!r} not found beside "
            f"{config_path} (looked for {path})"
        ) from e
    except Exception:
        # Leaving a half-executed module in sys.modules would make the next
        # import of that name silently succeed against a broken object.
        del sys.modules[name]
        raise

    return module


def _register(module):
    """Call the legacy registration hook, if the module has one.

    A module is expected to call `manta.registerPhysicsCase` at import, the way
    a C++ case registers during static initialisation. Every example module in
    this repository instead defines `registerTransportSystems()` and waits to be
    asked, which is the convention PyManta established. Both are honoured.
    """
    hook = getattr(module, "registerTransportSystems", None)
    if callable(hook):
        hook()


def load_physics_modules(config_path, extra_modules=()):
    """Import every physics module the config names, for its registrations."""
    conf = _configuration_of(config_path)

    # The current directory first, so a case sitting next to the config file
    # works without being installed.
    if "" not in sys.path:
        sys.path.insert(0, "")

    names = list(extra_modules)
    from_config = conf.get("PythonModule")
    if from_config and from_config not in names:
        names.append(from_config)

    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError as e:
            raise SystemExit(
                f"manta: could not import physics module {name!r}: {e}"
            ) from e
        _register(module)

    if "PythonModuleFile" in conf:
        _register(_load_module_from_file(config_path, conf))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="manta", description="Run a MaNTA transport problem from a config file."
    )
    parser.add_argument("config", nargs="?", default="MaNTA.conf",
                        help="TOML configuration file (default: MaNTA.conf)")
    parser.add_argument("--module", "-m", action="append", default=[], metavar="MODULE",
                        help="import MODULE before running, for its physics-case "
                             "registrations; repeatable. Also read from the config's "
                             "PythonModule key.")
    args = parser.parse_args(argv)

    load_physics_modules(args.config, args.module)

    return manta.run(args.config)


if __name__ == "__main__":
    sys.exit(main())
