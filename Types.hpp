#ifndef TYPES_HPP
#define TYPES_HPP

#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */
#include <Eigen/Core>


using Index = Eigen::Index;
using Value = double;

using Position = double;
using Time = double;
using Values = std::vector<double>;

using Vector = Eigen::Matrix<realtype,Eigen::Dynamic,1>;
using Matrix = Eigen::Matrix<realtype,Eigen::Dynamic,Eigen::Dynamic>;
using MatrixWrapper = Eigen::Map<Matrix>;
using VectorWrapper = Eigen::Map<Vector>;

#endif // TYPES_HPP
