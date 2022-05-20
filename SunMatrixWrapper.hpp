#pragma once

#include <sundials/sundials_matrix.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

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
	//To Do
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
	.space = nullptr
};
