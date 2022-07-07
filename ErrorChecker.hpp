#pragma once

class ErrorChecker
{
	//Full static class which can handle all boutique error checking 
private:
	/* data */
public:
	//Sundials function for checking return vals
	static int check_retval(void *returnvalue, const char *funcname, int opt);
};

