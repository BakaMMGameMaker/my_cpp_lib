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

add_executable(s_flat_hash_map_u32_basic
    s_flat_hash_map/tests/s_flat_hash_map_u32_basic.cpp
)

# =====================================================
#                     Target 2: Benchmarks
# =====================================================
add_executable(s_flat_hash_map_bench_emplace_default
    s_flat_hash_map/bench/s_flat_hash_map_bench_emplace_default.cpp
)

add_executable(bench_default_and_noit
    s_flat_hash_map/bench/bench_default_and_noit.cpp
)

add_executable(bench_default_and_no_rehash
    s_flat_hash_map/bench/bench_default_and_no_rehash.cpp
)

target_compile_definitions(s_flat_hash_map_bench_emplace_default PRIVATE BENCHMARK_STATIC_DEFINE)
target_compile_options(s_flat_hash_map_bench_emplace_default PRIVATE
    -Wno-used-but-marked-unused
    -Wno-switch-default
    -Wno-unneeded-internal-declaration
    -Wno-unused-template
)

target_compile_definitions(bench_default_and_noit PRIVATE BENCHMARK_STATIC_DEFINE)
target_compile_options(bench_default_and_noit PRIVATE
    -Wno-used-but-marked-unused
    -Wno-switch-default
    -Wno-unneeded-internal-declaration
    -Wno-unused-template
)

target_compile_definitions(bench_default_and_no_rehash PRIVATE BENCHMARK_STATIC_DEFINE)
target_compile_options(bench_default_and_no_rehash PRIVATE
    -Wno-used-but-marked-unused
    -Wno-switch-default
    -Wno-unneeded-internal-declaration
    -Wno-unused-template
)


target_link_libraries(s_flat_hash_map_bench_emplace_default PRIVATE
    benchmark
    benchmark_main
    shlwapi
)

target_link_libraries(bench_default_and_noit PRIVATE
    benchmark
    benchmark_main
    shlwapi
)

target_link_libraries(bench_default_and_no_rehash PRIVATE
    benchmark
    benchmark_main
    shlwapi
)
