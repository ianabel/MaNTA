#include "Variable.hpp"

void Variable::resize(int size)
{
	delqKappaFuncs.resize(size);
	deluKappaFuncs.resize(size);
	delqSourceFuncs.resize(size);
	deluSourceFuncs.resize(size);
}