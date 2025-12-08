cmake_minimum_required(VERSION 3.20)
project(my_cpp_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------- 全局编译器选项 -----------------
add_compile_options(
    -Weverything
    -Werror

    # 合法关闭：标准库会大量报警
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-disabled-macro-expansion
    -Wno-reserved-identifier
    -Wno-global-constructors
    -Wno-exit-time-destructors
    -Wno-unsafe-buffer-usage
    -Wno-padded
    -Wno-newline-eof

    -mavx2
    -D__AVX2__
)

# ----------------- include 路径 -----------------
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/s_flat_hash_map
    ${CMAKE_SOURCE_DIR}/s_trie
)

# ----------------- Google Benchmark -----------------
include_directories(
    D:/allfolders/code/advanced/cpp/benchmark-main/include
)

link_directories(
    D:/allfolders/code/advanced/cpp/benchmark-main/build/src/Release
)

# =====================================================
#                     Target 1: Basic Tests
# =====================================================
add_executable(s_flat_hash_map_basic
    s_flat_hash_map/tests/s_flat_hash_map_basic.cpp
)

# =====================================================
#                     Target 2: Benchmarks
# =====================================================
add_executable(s_flat_hash_map_bench
    s_flat_hash_map/bench/s_flat_hash_map_bench.cpp
)

# =====================================================
#                 Target 3: Small Benchmarks
# =====================================================
add_executable(s_flat_hash_map_bench_small
    s_flat_hash_map/bench/s_flat_hash_map_bench_small.cpp
)

target_compile_definitions(s_flat_hash_map_bench_small PRIVATE BENCHMARK_STATIC_DEFINE)
target_compile_options(s_flat_hash_map_bench_small PRIVATE
    -Wno-used-but-marked-unused
    -Wno-switch-default
    -Wno-unneeded-internal-declaration
    -Wno-unused-template
)


target_link_libraries(s_flat_hash_map_bench_small PRIVATE
    benchmark
    benchmark_main
    shlwapi
)


target_compile_definitions(s_flat_hash_map_bench PRIVATE BENCHMARK_STATIC_DEFINE)
target_compile_options(s_flat_hash_map_bench PRIVATE
    -Wno-used-but-marked-unused
    -Wno-switch-default
)

target_link_libraries(s_flat_hash_map_bench PRIVATE
    benchmark
    benchmark_main
    shlwapi
)
