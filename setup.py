"""Shell out to CMake to build the extension.

Everything else about the package is declared in pyproject.toml. This file
exists only because the extension's build needs SUNDIALS, netCDF, Eigen and
autodiff located through CMake; duplicating that discovery in setuptools would
leave two build systems that have to be kept in agreement, and CMake is the one
CI and the standalone binary already use.

The build directory is reused if it is already configured, so a checkout whose
`build/` cache already names SUNDIALS_ROOT (or whatever else this machine needs)
does not have to repeat it here -- the same role Makefile.local used to play.
Point MANTA_CMAKE_BUILD_DIR somewhere else to use a different one, and pass extra
configure arguments through MANTA_CMAKE_ARGS.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).parent.resolve()


class CMakeBuildPy(build_py):
    def run(self):
        # SOURCE_DATE_EPOCH-style opt-out, for a rebuild-free reinstall.
        # MANTA_SKIP_MAKE is the name this had before the move to CMake; it is
        # still honoured so an existing script or CI job does not break silently.
        if os.environ.get("MANTA_SKIP_BUILD") or os.environ.get("MANTA_SKIP_MAKE"):
            print("MANTA_SKIP_BUILD set; using the existing extension", file=sys.stderr)
        else:
            self.build_extension()

        built = list((HERE / "python" / "manta").glob("_manta*.so"))
        if not built:
            raise SystemExit(
                "The CMake build did not produce python/manta/_manta*.so.\n"
                "Check the configure output -- see README.md for first-time setup."
            )
        super().run()

    def build_extension(self):
        build_dir = Path(os.environ.get("MANTA_CMAKE_BUILD_DIR", HERE / "build"))
        extra = shlex.split(os.environ.get("MANTA_CMAKE_ARGS", ""))

        # Python3_EXECUTABLE is the important one: it makes the module's ABI
        # suffix and its headers come from the interpreter pip is running under,
        # rather than from whatever the build directory was last configured for.
        # Getting that wrong builds a module the installing interpreter cannot
        # import, and the error arrives later and somewhere else.
        configure = [
            "cmake",
            "-S", str(HERE),
            "-B", str(build_dir),
            f"-DPython3_EXECUTABLE={sys.executable}",
            *extra,
        ]
        if not (build_dir / "CMakeCache.txt").exists():
            # Only when creating the directory. Reusing one the developer
            # configured must not quietly rewrite their settings -- turning their
            # unit tests off as a side effect of `pip install .` would be a
            # surprising thing for a packaging step to do. MANTA_TESTS=OFF is
            # here so a machine with no Boost can still install the package.
            configure += ["-DCMAKE_BUILD_TYPE=Release", "-DMANTA_TESTS=OFF"]

        subprocess.run(configure, cwd=HERE, check=True)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "_manta",
             "-j", str(os.cpu_count() or 1)],
            cwd=HERE, check=True,
        )


setup(cmdclass={"build_py": CMakeBuildPy})
