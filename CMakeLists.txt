cmake_minimum_required(VERSION 3.12)
project(MaNTA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DDEBUG -g -O0 -march=native -Wall -Wno-parentheses -Werror -Wno-error=unused-variable -pedantic")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(MaNTA_BUILD_UNIT_TESTS "Build MaNTA unit tests." on)
option(MaNTA_BUILD_PHYSICS_CASES "Build all MaNTA physics cases." on)

set(SOURCES main.cpp MaNTA.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp AutodiffFlux.cpp)
set(HEADERS gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp TransportSystem.hpp PhysicsCases.hpp DGSoln.hpp AutodiffFlux.hpp)

set(PHYSICS_SOURCES 3VarCylFlux.cpp AutodiffTransportSystem.cpp Constants.cpp FishersEquation.cpp FourVarCylFlux.cpp LD2.cpp LinearDiffusion.cpp MatrixDiffusion.cpp MatrixDiffusionTest.cpp MatrixFlux.cpp NonlinearDiffusion.cpp)
set(PHYSICS_HEADERS 3VarCylFlux.hpp AutodiffTransportSystem.hpp Constants.hpp FishersEquation.hpp FourVarCylFlux.hpp LD2.hpp LinearDiffusion.hpp MatrixDiffusion.hpp MatrixDiffusionTest.hpp MatrixFlux.hpp NonlinearDiffusion.hpp)

if (MaNTA_BUILD_PHYSICS_CASES)
    foreach(case IN LISTS PHYSICS_SOURCES)
        list(APPEND SOURCES PhysicsCases/${case})
    endforeach()

    foreach(case IN LISTS PHYSICS_HEADERS)
        list(APPEND HEADERS PhysicsCases/${case})
    endforeach()
endif()

set(AUTODIFF_DIR "/home/eatocco/packages/autodiff")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Boost REQUIRED)
find_package(SUNDIALS MODULE)
find_package(NetCDF MODULE)
find_package(autodiff CONFIG PATHS ${AUTODIFF_DIR})
find_package(Eigen3 CONFIG)

add_executable(MaNTA ${SOURCES} ${HEADERS})

add_subdirectory(toml11/)

target_link_libraries(MaNTA toml11)

if(Boost_FOUND)
    message("Boost Found")
    target_include_directories(MaNTA PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(MaNTA ${Boost_LIBRARIES})
elseif(NOT Boost_FOUND)
    message("Boost Not Found")
endif()

if(autodiff_FOUND)
    message("autodiff Found")
    target_include_directories(MaNTA PRIVATE ${autodiff_DIR}/../../../include)
    target_link_libraries(MaNTA ${autodiff_LIBRARIES})
elseif(NOT autodiff_FOUND)
    message("autodiff Not Found")
endif()

if(SUNDIALS_FOUND)
    message("SUNDIALS Found")
    target_include_directories(MaNTA PRIVATE "${SUNDIALS_INCLUDE_DIRS}")
    target_link_libraries(MaNTA ${SUNDIALS_LIBRARIES})
elseif(NOT SUNDIALS_FOUND)
    message("SUNDIALS Not Found")
endif()

if(Eigen3_FOUND)
    message("Eigen3 Found")
    target_include_directories(MaNTA PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(MaNTA ${EIGEN3_LIBRARIES})
elseif(NOT Eigen3_FOUND)
    message("Eigen Not Found")
endif()

if(NetCDF_FOUND)
    message("NetCDF Found")
    list(APPEND NetCDF_LIBRARIES "netcdf_c++4")
    target_include_directories(MaNTA PRIVATE ${NetCDF_INCLUDE_DIRS})
    target_link_libraries(MaNTA ${NetCDF_LIBRARIES})
elseif(NOT NetCDF_FOUND)
    message("NetCDF Not Found")
endif()






