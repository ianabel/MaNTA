#include "SunMatrixWrapper.hpp"

SUNMatrix_ID MatGetID( SUNMatrix mat)
{
	return SUNMATRIX_CUSTOM;
}

int MatZero( SUNMatrix mat )
{
	return 0;
}

void MatDestroy( SUNMatrix mat  )
{
	delete reinterpret_cast<SunMatrixWrapper*>( mat->content );
	mat->content = nullptr;
	SUNMatFreeEmpty( mat );
}

SUNMatrix SunMatrixNew(SUNContext ctx)
{
	// As SunLinSolWrapper::SunLinSol: the operations this matrix provides, on
	// the table SUNMatNewEmpty allocated, so a member SUNDIALS adds or drops
	// (`space`, in 8.0) changes nothing here.
	SUNMatrix mat = SUNMatNewEmpty(ctx);
	mat->content = new SunMatrixWrapper();
	mat->ops->getid = MatGetID;
	mat->ops->destroy = MatDestroy;
	mat->ops->zero = MatZero;
	return mat;
}