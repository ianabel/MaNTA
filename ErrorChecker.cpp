#include "ErrorChecker.hpp"
#include <print>

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */
int ErrorChecker::check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    std::print(stderr, "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
               funcname);
    return(1);
  } else if (opt == 1) {
    /* Check if retval < 0 */
    retval = (int *) returnvalue;
    if (*retval < 0) {
      std::print(stderr, "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
                 funcname, *retval);
      return(1);
    }
  } else if (opt == 2 && returnvalue == NULL) {
    /* Check if function returned NULL pointer - no memory allocated */
    std::print(stderr, "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
               funcname);
    return(1);
  }

  return(0);
}