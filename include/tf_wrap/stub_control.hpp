#pragma once

#include <tensorflow/c/c_api.h>

namespace tf_wrap::stub {

#if TF_WRAPPER_USE_STUB
extern "C" void TF_StubSetNextError(const char* api, TF_Code code, const char* message);
#endif

} // namespace tf_wrap::stub
