#ifndef SUNMATRIXWRAPPER_HPP
#define SUNMATRIXWRAPPER_HPP

#include <sundials/sundials_matrix.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>        /* defs of realtype, sunindextype  */

//this sun matrix template is empty by design. It provides nothing to the linear solver.
//All matrix data is stored within the coefficients of the DGApprox objects
//This is provided to the LS so that sundials doesn't assume a matrixless solver which would require an iterative solver.
//We solve the jacobian equation of the system exactly so no iteration is required to generate an approximate Jacobian.
class SunMatrixWrapper
{
public:
	SunMatrixWrapper() = default;
	~SunMatrixWrapper() = default;
private:
};

SUNMatrix_ID MatGetID(SUNMatrix mat);

int MatZero(SUNMatrix mat);

void MatDestroy( SUNMatrix mat  );

SUNMatrix SunMatrixNew(SUNContext ctx);


#endif // SUNMATRIXWRAPPER_HPP
