#pragma once

#include <sundials/sundials_matrix.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

//this sun matrix template is purposely empty. It provides nothing to the linear solver.
//All matrix data is stored within the coefficients of the DGApprox objects
//This is provided to the LS so that sundials doesn't assume a matrixless solver which would require an iterative solver.
//We can solve the system exactly so not iteration is required.
class SunMatrixWrapper
{
public:
	SunMatrixWrapper() = default;
	~SunMatrixWrapper() = default;
private:
};

SUNMatrix_ID MatGetID( SUNMatrix mat)
{
	return SUNMATRIX_CUSTOM;
}

int MatZero( SUNMatrix mat)
{
	return 0;
}

struct _generic_SUNMatrix_Ops MatOps {
	.getid = MatGetID,
	.clone = nullptr,
	.destroy = nullptr,
	.zero = MatZero,
	.copy = nullptr,
	.scaleadd = nullptr,
	.scaleaddi = nullptr,
	.matvecsetup = nullptr,
	.matvec = nullptr,
	.space = nullptr,
};

SUNMatrix SunMatrixNew()
{
	SUNMatrix mat = SUNMatNewEmpty();
	mat->content = new SunMatrixWrapper();
	mat->ops = &MatOps;
	return mat;

}
