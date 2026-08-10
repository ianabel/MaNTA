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
"""

import argparse
import importlib
import sys
import tomllib

import manta


def _python_module_of(config_path):
    """The PythonModule key, if the config has one."""
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        # Let runManta report it, so the message is the same either way.
        return None
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"manta: {config_path} is not valid TOML: {e}") from e

    return config.get("configuration", {}).get("PythonModule")


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

    modules = list(args.module)
    from_config = _python_module_of(args.config)
    if from_config and from_config not in modules:
        modules.append(from_config)

    # The current directory first, so a case sitting next to the config file
    # works without being installed.
    if "" not in sys.path:
        sys.path.insert(0, "")

    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as e:
            raise SystemExit(f"manta: could not import physics module {name!r}: {e}") from e

    return manta.run(args.config)


if __name__ == "__main__":
    sys.exit(main())
