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
	mat->ops = nullptr;
	SUNMatFreeEmpty( mat );
}

struct _generic_SUNMatrix_Ops MatOps {
	.getid = MatGetID,
	.clone = nullptr,
	.destroy = MatDestroy,
	.zero = MatZero,
	.copy = nullptr,
	.scaleadd = nullptr,
	.scaleaddi = nullptr,
	.matvecsetup = nullptr,
	.matvec = nullptr,
	.space = nullptr,
};

SUNMatrix SunMatrixNew(SUNContext ctx)
{
	SUNMatrix mat = SUNMatNewEmpty(ctx);
	mat->content = new SunMatrixWrapper();
	mat->ops = &MatOps;
	return mat;
}