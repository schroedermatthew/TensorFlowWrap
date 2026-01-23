#pragma once

// ============================================================================
// TF_TString stub for testing without TensorFlow
// ============================================================================

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// TF_TString - String tensor element type
typedef struct TF_TString {
    char* data;
    size_t size;
    size_t capacity;
} TF_TString;

// TF_TString functions
void TF_TString_Init(TF_TString* tstr);
void TF_TString_Copy(TF_TString* dst, const char* src, size_t size);
const char* TF_TString_GetDataPointer(const TF_TString* tstr);
size_t TF_TString_GetSize(const TF_TString* tstr);

#ifdef __cplusplus
}
#endif
