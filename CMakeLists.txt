cmake_minimum_required(VERSION 3.0.0)
project(MaNTA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(SOURCES MTS.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp AutodiffFlux.cpp)
set(HEADERS gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp TransportSystem.hpp PhysicsCases.hpp DGSoln.hpp AutodiffFlux.hpp)
add_executable(MaNTa ${SOURCES})

add_subdirectory(PhysicsCases/)

option(MaNTa_BUILD_UNIT_TESTS "Build MaNTa unit tests." on)

set(AUTODIFF_DIR "/home/eatocco/autodiff")
set(SUNDIALS_INCLUDE_DIR "/home/eatocco/sundials/lib/")
set(SUNDIALS_LIB_DIR "/home/eatocco/sundials/lib/lib")
set(EIGEN_DIR "/usr/include/eigen3")

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "/usr/")

find_package(Boost REQUIRED)


add_subdirectory(toml11/)

target_link_libraries(MaNTa toml11)

include_directories(${SUNDIALS_INCLUDE_DIR})
find_library(SUNDIALS_LIB SUNDIALS PATHS ${SUNDIALS_LIB_DIR})
message(STATUS "MY_CAMERA_LIBRARY: " ${SUNDIALS_LIB})
target_link_libraries(MaNTa ${SUNDIALS_LIB})

include_directories(${EIGEN_DIR})
#target_link_libraries(MaNTa Eigen)

include_directories(${AUTODIFF_DIR})
link_directories(MaNTa INTERFACE ${AUTODIFF_DIR}/lib)

include_directories("/usr/include")
target_link_libraries(MaNTa netcdf)

if(Boost_FOUND)
    message("Boost Found")
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(MaNTa ${Boost_LIBRARIES})
elseif(NOT Boost_FOUND)
    error("Boost Not Found")
endif()



