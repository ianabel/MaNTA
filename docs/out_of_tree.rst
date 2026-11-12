Working outside the MaNTA tree
==============================

A physics case and the driver that runs it do not have to live in this
repository. There are two routes, one for each language, and they can be mixed:
a Python driver can run a C++ plugin and vice versa.

Python: install the package
---------------------------

.. code-block:: sh

   cmake --build build --target _manta   # the package, at build/python/manta
   pip install .        # installs the `manta` package and the `manta` command
   pip install .[jax]   # ...and the JAX layer, if you want manta.jax

``python-examples/`` is the worked version of everything in this section. Each
directory there holds a case, its config and a README, and imports ``manta``
exactly as your own package would — copy one out of the repository and it still
runs. ``python-examples/linear-diffusion`` is the place to start.

Then a case in your own package imports ``manta`` like any other dependency.
This one, and the config below, are the files CI's out-of-tree job installs the
package for and runs (``Tests/OutOfTree/``):

.. literalinclude:: ../Tests/OutOfTree/mypackage/mycase.py
   :language: python
   :caption: mypackage/mycase.py
   :lines: 5-

The ``config`` the factory hands a case is the whole parsed file, as a C++ case
gets; the case reads its own table from it.

Only ``SigmaFn`` and ``Sources`` are required. The derivative hooks are
optional, and an omitted one means that block is identically zero — which is
why the case above writes ``dSigmaFn_dq`` and nothing else.

Run it with the ``manta`` command, naming the module that registers it:

.. literalinclude:: ../Tests/OutOfTree/run.conf
   :language: toml
   :caption: run.conf

.. code-block:: sh

   manta run.conf

The module is imported for its side effects — it is expected to call
``registerPhysicsCase`` at import, exactly as a C++ case registers itself during
static initialisation — and control then passes to the same solver the
standalone binary uses. ``manta --module NAME`` does the same from the command
line, and is repeatable.

.. note::

   The wheel is not portable: the extension links the SUNDIALS, netCDF and BLAS
   that CMake found. ``pip install .`` shells out to CMake rather than
   reimplementing that discovery, so the build needs a working :doc:`install`. It
   reuses ``build/`` if that is already configured, so a machine whose SUNDIALS
   needed naming does not have to name it again for the install; point
   ``MANTA_CMAKE_BUILD_DIR`` elsewhere, or pass extra configure arguments through
   ``MANTA_CMAKE_ARGS``, if you want something different.

C++: build a plugin
-------------------

.. code-block:: sh

   cmake --install build --prefix /where/you/want

installs the headers under ``$PREFIX/include/manta``, ``libmanta.so`` under
``$PREFIX/lib``, and a pkg-config file recording that prefix. (There is
deliberately no installed CMake package: an exported target would invite
``target_link_libraries(mycase manta)``, and linking ``libmanta`` is precisely
the mistake that gives a plugin its own copy of the registry.) A physics case is then an ordinary
shared object:

.. literalinclude:: ../Tests/OutOfTree/MyCase.cpp
   :language: cpp
   :caption: MyCase.cpp
   :lines: 7-

.. code-block:: sh

   g++ -shared -fPIC $(pkg-config --cflags manta) MyCase.cpp -o libmycase.so

and name it in the config:

.. literalinclude:: ../Tests/OutOfTree/plugin.conf
   :language: toml
   :caption: plugin.conf

and run it with the installed binary, ``$PREFIX/bin/MaNTA plugin.conf``.
``Tests/OutOfTree/run-out-of-tree.sh`` does all of this -- install, compile,
run, and the ``pip install .`` route above -- and is what CI runs.

The solver ``dlopen``\ s each entry before instantiating the problem. The
plugin's static initialiser runs on load and registers the case into the same
process-global map the built-in cases use.

From Python, ``manta.load_physics_plugin("./libmycase.so")`` does the same thing,
and then ``manta.Runner("MyCase")`` reaches the case — see
:ref:`running-cpp-cases`. ``PhysicsPlugins`` itself is not accepted in a
``configure`` dict, for the same reason ``TransportSystem`` is not: choosing the
physics is not a solver setting.

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
   ``-DEIGEN_USE_BLAS`` and ``-DEIGEN_DONT_PARALLELIZE`` are carried for the
   same reason: each changes the code Eigen's templates expand to, and both
   sides of the boundary must expand them alike.

   *Linking the library.* The solver links the core objects directly, so a
   plugin that also pulled in ``libmanta.so`` would get a **second** copy of
   ``PhysicsCases::map`` and register itself into a map the solver never reads —
   silently, with the case simply missing at run time. Compile against the
   headers alone and leave the MaNTA symbols undefined; the loader binds them to
   the host process, which is linked ``-rdynamic`` for exactly this.
   ``libmanta.so`` is for embedding the solver in another program, a different
   job.
