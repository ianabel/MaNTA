#include "Variable.hpp"

void Variable::resize(int size)
{
	delqKappaFuncs.resize(size);
	deluKappaFuncs.resize(size);
	delsigSourceFuncs.resize(size);
	delqSourceFuncs.resize(size);
	deluSourceFuncs.resize(size);
}