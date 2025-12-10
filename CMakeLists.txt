cmake_minimum_required(VERSION 3.20)
project(my_cpp_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =====================================================
#                    Global compile options
# =====================================================
add_compile_options(
    -Weverything
    -Werror

    # 防止标准库关闭报警
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-disabled-macro-expansion
    -Wno-reserved-identifier
    -Wno-global-constructors
    -Wno-exit-time-destructors
    -Wno-unsafe-buffer-usage
    -Wno-padded
    -Wno-newline-eof

    # -mavx2
    # -D__AVX2__
)

# =====================================================
#                    include 路径
# =====================================================
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/s_hash_map
    ${CMAKE_SOURCE_DIR}/s_trie
)

# =====================================================
#                  Google Benchmark
# =====================================================
include_directories(
    D:/allfolders/code/advanced/cpp/benchmark-main/include
)

link_directories(
    D:/allfolders/code/advanced/cpp/benchmark-main/build/src/Release
)

# =====================================================
#                     Target 1: Tests
# =====================================================
add_executable(s_flat_hash_map_basic
    s_hash_map/tests/s_flat_hash_map_basic.cpp
)

add_executable(test_fmu32
    s_hash_map/tests/test_fmu32.cpp
)

# =====================================================
#                Benchmark Target Helper
# =====================================================

# 所有 benchmark 源文件
set(BENCH_TARGETS
    bench_default_vs_fast
    bench_default_vs_no_check_dup
    bench_default_vs_no_rehash
    bench_default_vs_no_ret
    bench_fmu32_vs_std_find_exist
    bench_fmu32_vs_std_insert_random
    bench_fmu32_vs_std_insert_unique
    bench_fmu32_vs_std
    bench_vs_std
)

# benchmark 的公共 compile 定义
set(BENCH_COMMON_DEFS
    BENCHMARK_STATIC_DEFINE
)

# benchmark 的公共 warnings
set(BENCH_COMMON_OPTS
    -Wno-used-but-marked-unused
    -Wno-switch-default
    -Wno-unneeded-internal-declaration
    -Wno-unused-template
)

# benchmark 公共 link libs
set(BENCH_COMMON_LIBS
    benchmark
    benchmark_main
    shlwapi
)

# =====================================================
#                     Target 2: Benchmarks
# =====================================================
foreach(tgt IN LISTS BENCH_TARGETS)
    add_executable(${tgt}
        s_hash_map/bench/${tgt}.cpp
    )

    target_compile_definitions(${tgt} PRIVATE ${BENCH_COMMON_DEFS})
    target_compile_options(${tgt} PRIVATE ${BENCH_COMMON_OPTS})
    target_link_libraries(${tgt} PRIVATE ${BENCH_COMMON_LIBS})
endforeach()
