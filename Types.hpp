#ifndef TYPES_HPP
#define TYPES_HPP

#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>


using Index = Eigen::Index;
using Value = double;

using Vector = Eigen::Matrix<double,Eigen::Dynamic,1>;
using Matrix = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
using MatrixWrapper = Eigen::Map<Matrix>;
using VectorWrapper = Eigen::Map<Vector>;

using Position = double;
using Time = double;
using Values = Vector;

#endif // TYPES_HPP
