// compile_fail_status_copy.cpp
// This file should FAIL to compile.
// Tests that Status is non-copyable (deleted copy constructor).
//
// Expected error: use of deleted function / copy constructor is deleted

#include "tf_wrap/status.hpp"

int main() {
    tf_wrap::Status s1;
    
    // Status is non-copyable - this should fail to compile
    tf_wrap::Status s2 = s1;
    (void)s2;
    return 0;
}
