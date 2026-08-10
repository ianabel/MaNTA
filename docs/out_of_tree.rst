Working outside the MaNTA tree
==============================

A physics case and the driver that runs it do not have to live in this
repository. There are two routes, one for each language, and they can be mixed:
a Python driver can run a C++ plugin and vice versa.

Python: install the package
---------------------------

.. code-block:: sh

   make python          # builds python/manta/_manta<abi>.so
   pip install .        # installs the `manta` package and the `manta` command

Then a case in your own package imports ``manta`` like any other dependency:

.. code-block:: python

   # mypackage/mycase.py
   import manta
   import numpy as np

   class MyDiffusion(manta.TransportSystem):
       variables = [manta.Field("temperature", "electron temperature", "eV")]

       def __init__(self, config, grid):
           super().__init__()
           self.kappa = config["Kappa"]

       def SigmaFn(self, i, state, x, t):
           return self.kappa * state["Derivative"][0]

       def Sources(self, i, state, x, t):
           return 1.0

       def dSigmaFn_dq(self, i, state, x, t):
           return np.full(self.nVars, self.kappa)

       def LowerBoundary(self, i, t): return 0.0
       def UpperBoundary(self, i, t): return 0.0
       def InitialValue(self, i, x): return 0.0
       def InitialDerivative(self, i, x): return 0.0

   manta.registerPhysicsCase("MyDiffusion", MyDiffusion)

Only ``SigmaFn`` and ``Sources`` are required. The derivative hooks are
optional, and an omitted one means that block is identically zero — which is
why the case above writes ``dSigmaFn_dq`` and nothing else.

Run it with the ``manta`` command, naming the module that registers it:

.. code-block:: toml

   [configuration]
   TransportSystem = "MyDiffusion"
   PythonModule = "mypackage.mycase"
   Polynomial_degree = 2
   Grid_size = 8
   Lower_boundary = 0.0
   Upper_boundary = 1.0
   delta_t = 0.05
   t_final = 0.2

.. code-block:: sh

   manta run.conf

The module is imported for its side effects — it is expected to call
``registerPhysicsCase`` at import, exactly as a C++ case registers itself during
static initialisation — and control then passes to the same solver the
standalone binary uses. ``manta --module NAME`` does the same from the command
line, and is repeatable.

.. note::

   The wheel is not portable: the extension links the SUNDIALS, netCDF and BLAS
   that ``Makefile.local`` pointed at. ``pip install .`` shells out to
   ``make python`` rather than reimplementing that discovery, so the build needs
   a working :doc:`install`.

C++: build a plugin
-------------------

.. code-block:: sh

   make install PREFIX=/where/you/want

installs the headers under ``$PREFIX/include/manta``, ``libmanta.so`` under
``$PREFIX/lib``, and a pkg-config file. A physics case is then an ordinary
shared object:

.. code-block:: cpp

   // MyCase.cpp
   #include <manta/PhysicsCases.hpp>

   class MyCase : public TransportSystem
   {
   public:
       MyCase(toml::value const &config, Grid const &)
           : TransportSystem({.variables = {{"u", "the diffused quantity", "kg/m^3"}}})
       {
       }

       Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
       // ... the rest of the interface

   private:
       REGISTER_PHYSICS_HEADER(MyCase)
   };

   REGISTER_PHYSICS_IMPL(MyCase);

.. code-block:: sh

   g++ -shared -fPIC $(pkg-config --cflags manta) MyCase.cpp -o libmycase.so

and name it in the config:

.. code-block:: toml

   [configuration]
   TransportSystem = "MyCase"
   PhysicsPlugins = [ "./libmycase.so" ]

The solver ``dlopen``\ s each entry before instantiating the problem. The
plugin's static initialiser runs on load and registers the case into the same
process-global map the built-in cases use.

.. warning::

   **Compile the plugin with the flags pkg-config reports, and do not link
   ``-lmanta``.**

   Two things will bite otherwise, and neither is a link error.

   *Wrong architecture flags.* Eigen aligns its types to the widest vector unit
   the compiler is told about, and inlines its expression templates into both
   sides of the boundary. A plugin built without the ``-march=`` the core was
   built with lays out and loads Eigen objects differently, and the symptom is a
   ``SIGSEGV`` inside an aligned AVX-512 load the first time the solver touches
   the plugin's state. This is why ``manta.pc`` records the concrete
   architecture rather than ``native``: a plugin compiled on a different machine
   is then rejected by the compiler instead of crashing at run time.
   ``-DEIGEN_USE_BLAS`` is carried for the same reason.

   *Linking the library.* The solver links the core objects directly, so a
   plugin that also pulled in ``libmanta.so`` would get a **second** copy of
   ``PhysicsCases::map`` and register itself into a map the solver never reads —
   silently, with the case simply missing at run time. Compile against the
   headers alone and leave the MaNTA symbols undefined; the loader binds them to
   the host process, which is linked ``-rdynamic`` for exactly this.
   ``libmanta.so`` is for embedding the solver in another program, a different
   job.
