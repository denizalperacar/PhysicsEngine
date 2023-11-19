#ifndef F60A8C91_EC5C_405A_B4AB_C8CEDFE056A6
#define F60A8C91_EC5C_405A_B4AB_C8CEDFE056A6


#include "../common/common.h"

PE_BEGIN

#define cudaErr(x) fmt::print("CUDA ERR CHECK:\n    {} {}\n", x, cudaGetErrorString(cudaGetLastError()));

PE_END

#endif /* F60A8C91_EC5C_405A_B4AB_C8CEDFE056A6 */
