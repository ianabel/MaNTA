"""Shell out to the project Makefile to build the extension.

Everything else about the package is declared in pyproject.toml. This file
exists only because the extension's build needs SUNDIALS, netCDF, Eigen and
autodiff located through Makefile.local; duplicating that discovery in
setuptools would leave two build systems that have to be kept in agreement, and
the Makefile is the one CI and the standalone binary already use.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).parent.resolve()


class MakeBuildPy(build_py):
    def run(self):
        # SOURCE_DATE_EPOCH-style opt-out, for a rebuild-free reinstall.
        if os.environ.get("MANTA_SKIP_MAKE"):
            print("MANTA_SKIP_MAKE set; using the existing extension", file=sys.stderr)
        else:
            subprocess.run(["make", "python"], cwd=HERE, check=True)

        built = list((HERE / "python" / "manta").glob("_manta*.so"))
        if not built:
            raise SystemExit(
                "make python did not produce python/manta/_manta*.so.\n"
                "Check Makefile.local -- see README.md for first-time setup."
            )
        super().run()


setup(cmdclass={"build_py": MakeBuildPy})
