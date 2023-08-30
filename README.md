# MaNTA -- The Maryland Nonliner Transport Analyzer

MaNTA is a code developed for the prediction and analysis of transport processes in a strongly-magnetized plasma.
MaNTA solves

## Getting Started

You will need to download this codebase and compile it in order to run MaNTA

### Prerequisites

To compile and use MaNTA you will need a system with the following

 - A C++20 compliant C++ compiler (tested on g++ 13 and clang++ 16).
 - The Boost C++ Template Library
 - The TOML11 library
 - The SUNDIALS library, Version 6.0.0 or newer
 - NETCDF C and NETCDF C++ 4.3 or newer (depends upon netcdf C interface 4.6.0 or newer)
 - The Autodiff C++ library

Precise dependencies have not been exhaustively tested, bug reports are welcome. Running on Windows requires the installation of [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) (WSL) 
and proceeding as for linux.

MacOS has yet to be tested.

### Building MaNTA

All the build options are set in the file `Makefile.local`, which you need to provide for your system.
An example is provided in `Makefile.local.example` -- copy this file to `Makefile.local` and make any edits needed.
This file is in GNU-compatable Makefile format, and you can set and override all the compilation options here.
For example, if you are not using the default compiler (g++), then you can add a line to `Makefile.local` that reads `CXX = /path/to/my/c++/compiler`.

If you're happy with this, let's proceed!

 1. Clone this repository into your chosen location.
 2. Install the Boost library, either using your system package manager or manually by downloading from [here](https://www.boost.org). If this is a system-wide install,
 proceed to step 3. If you downloaded the Boost libraries, add a line to `Makefile.local` which sets `BOOST_DIR = /path/to/boost`.
 3. Clone the [TOML11](http://github.com/toruniina/toml11) library into a directory of your choice. If you clone it into the default location of MCTrans/toml11, proceed to step 3. As with Boost, set `TOML11_DIR = /path/to/toml11` in `Makefile.local`.
 4. Install [SUNDIALS](https://computing.llnl.gov/projects/sundials) and edit Makefile.local to set `SUNDIALS_DIR` to the location you have installed the Sundials library in. If you are only using SUNDIALS for MCTrans++, a quick intro to installing SUNDIALS is inclued below.
 5. Install [NETCDF C and NETCDF C++](https://www.unidata.ucar.edu/software/netcdf/). On Ubuntu or Debian these can be installed from the package manager: `apt-get install libnetcdf-dev libnetcdff-dev libnetcdf-c++4-dev libnetcdf-c++4-1`. 
 On MacOS, you can use either `brew install netcdf` or `conda install -c anaconda netcdf4` to install the C version, and `conda install -c conda-forge netcdf-cxx4` to install the C++ version. 
 Please specify in `Makefile.local` where these libraries are installed. For example, `NETCDF_DIR = /usr/local/Cellar/netcdf/4.8.0_2` and `NETCDF_CXX_DIR = /Users/<username>/miniconda3` if you used `brew` and `conda` to install on MacOS.
 6. Set any other options, e.g. setting the variable `DEBUG` to any value will build a version that you can use to develop MCTrans++ and that includes debug information.
 7. Run `make`.
 8. Check the unit tests with `make test`. 

#### Installing SUNDIALS

If you are only building a version of SUNDIALS for use with MaNTA the included script `build_sundials` should provide
the minimal needed installation of SUNDIALS. If using MacOS, `coreutils` and `cmake` must be installed to run the build script.

If this is your first use of SUNDIALS, and you want a custom install, a quick guide to installing the base libraries follows here.

Pick where you want the sundials sources / build tree / compiled libraries to go. We will call these directories
SUNDIALS_SOURCE, SUNDIALS_BUILD, and SUNDIALS_INSTALL in the following. One suggestion would be
```
SUNDIALS_SOURCE = ~/sundials/source
SUNDIALS_BUILD  = ~/sundials/build
SUNDIALS_INSTALL = ~/sundials/
```

With these directories picked, we can download and compile SUNDIALS.

 1. Download the SUNDIALS source from [here](https://computing.llnl.gov/projects/sundials) or [here](https://github.com/LLNL/sundials) into `SUNDIALS_SOURCE`
 2. Move to `SUNDIALS_BUILD`. Configure the SUNDIALS build with
 ```
 cmake $SUNDIALS_SOURCE -DCMAKE_INSTALL_PREFIX=$SUNDIALS_INSTALL -DEXAMPLES_INSTALL=off
 ```
	   If this gives you any errors (lack of C compiler, etc), refer to the SUNDIALS documentation.
 3. Compile SUNDIALS with `make -j install`.
 4. You now have sundials installed into the `SUNDIALS_INSTALL` directory. This is the path you should set `SUNDIALS_DIR` to in your MCTrans `Makefile.local`

### Providing new Transport Systems

The core MaNTA algorithm solves a generic set of reaction-diffusion equations. The physics models are all contained in the `PhysicsCases/` directory.

### Example Configurations

Example configurations live in the `examples/` subdirectory. 

## Built With

* [Boost](http://boost.org) - C++ Template library that radically extends the STL
* [TOML11](http://github.com/toruniina/toml11) - For parsing configuration files written in [TOML](https://github.com/toml-lang/toml)
* [Sundials](https://computing.llnl.gov/projects/sundials) - Suite of libraries from Lawrence Livermore National Laboratory for numerical solution of Nonlinear Algebraic Equations, ODEs and DAEs
* [NETCDF C and NETCDF C++](https://www.unidata.ucar.edu/software/netcdf/) - A set of software libraries and machine-independent data formats that support the creation, access, and sharing of array-oriented scientific data.

## Known Issues


Specific known issues are listed here.


## Contributing

Contributions to this project through the github interface are welcome. Please email the authors to help out

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/ianabel/MaNTA/tags).

## Authors

* **Myles Kelly** - *Original author*
* **Ian Abel** - *Maintainer* - [Ian Abel at UMD](https://ireap.umd.edu/faculty/abel)
* **Eddie Tocco** - *Autodiff interfcaces*

For full copyright attribution, see the [COPYRIGHT](COPYRIGHT) file.
For a summary of contributors, see the [contributors](http://github.com/ianabel/MCTrans/contributors) page.

## License

This project is licensed under the 3-Clause BSD Licence - see the [LICENSE.md](LICENSE.md) file for details
