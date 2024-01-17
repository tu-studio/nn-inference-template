#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
// #include <cstring> // necessary for strerror
// #include <stdexcept>
// #include <sys/mman.h> // necessary for mlockall

// void LockMemory() {
//     int ret = mlockall(MCL_CURRENT | MCL_FUTURE);
//     if (ret) {
//         throw std::runtime_error{std::strerror(errno)};
//     }
// }

TEST(Benchmark, All){
    // LockMemory();
    benchmark::RunSpecifiedBenchmarks();
    std::cout << "--------------------------------------------------------------" << std::endl;
}