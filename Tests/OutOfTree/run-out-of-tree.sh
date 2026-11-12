#!/bin/sh
# Build and run the two out-of-tree examples the way docs/out_of_tree.rst tells a
# user to, each from a directory outside the checkout:
#
#   * a C++ case, compiled against an installed MaNTA with pkg-config's flags
#     and loaded through PhysicsPlugins by the installed binary;
#   * a Python case in a package of its own, run by the `manta` command from a
#     `pip install .` into a fresh virtualenv.
#
# Usage: run-out-of-tree.sh BUILD_DIR WORK_DIR
#
# BUILD_DIR is a configured and built MaNTA build directory; WORK_DIR is created
# and holds everything this writes -- the install prefix, the virtualenv and both
# runs. `pip install .` reuses BUILD_DIR (MANTA_CMAKE_BUILD_DIR) and reconfigures
# it for the virtualenv's interpreter, as setup.py always does. $CXX must be the
# compiler BUILD_DIR was built with, $PYTHON the interpreter to make the
# virtualenv from.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../.." && pwd)
build=$(cd "$1" && pwd)
mkdir -p "$2"
work=$(cd "$2" && pwd)
prefix="$work/prefix"

echo "== cmake --install into $prefix"
cmake --install "$build" --prefix "$prefix" > "$work/install.log"

echo "== a C++ plugin against the installed headers"
pc=$(find "$prefix" -name manta.pc | head -n 1)
test -n "$pc" || { echo "no manta.pc under $prefix" >&2; exit 1; }
export PKG_CONFIG_PATH="$(dirname "$pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
rm -rf "$work/plugin"
mkdir -p "$work/plugin"
cp "$here/MyCase.cpp" "$here/plugin.conf" "$work/plugin/"
cd "$work/plugin"
# shellcheck disable=SC2046 # pkg-config's output is a list of flags
${CXX:-c++} -shared -fPIC $(pkg-config --cflags manta) MyCase.cpp -o libmycase.so
"$prefix/bin/MaNTA" plugin.conf > run.log
grep -q "^Done" run.log || { cat run.log; echo "the plugin run did not finish" >&2; exit 1; }
test -s plugin.nc

echo "== a Python case, through pip install . and the manta command"
rm -rf "$work/venv" "$work/python"
"${PYTHON:-python3}" -m venv "$work/venv"
MANTA_CMAKE_BUILD_DIR="$build" "$work/venv/bin/pip" install --quiet "$src"
mkdir -p "$work/python"
cp -r "$here/mypackage" "$here/run.conf" "$work/python/"
cd "$work/python"
"$work/venv/bin/manta" run.conf > run.log
grep -q "^Done" run.log || { cat run.log; echo "the Python run did not finish" >&2; exit 1; }
test -s run.nc

echo "== both ran"
