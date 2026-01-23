// CMake integration test - verifies tf::wrapper target works after find_package
// This file should compile if the installed headers are correct

#include <tf_wrap/all.hpp>
#include <iostream>

int main() {
    // Test that basic types are available
    tf_wrap::Status status;
    status.set(TF_OK, "Integration test");
    
    if (status.ok()) {
        std::cout << "CMake integration test PASSED\n";
        std::cout << "  - Headers are accessible\n";
        std::cout << "  - tf::wrapper target works\n";
        return 0;
    }
    
    std::cout << "CMake integration test FAILED\n";
    return 1;
}
