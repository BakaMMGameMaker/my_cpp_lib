cmake_minimum_required(VERSION 3.20)
project(my_cpp_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------- 编译器选项 -----------------
add_compile_options(
    -Weverything
    -Werror

    # 合法关闭：标准库会疯狂报
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-disabled-macro-expansion
    -Wno-reserved-identifier
    -Wno-global-constructors
    -Wno-exit-time-destructors
    -Wno-unsafe-buffer-usage
    -Wno-padded
    -Wno-newline-eof
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

# ----------------- 测试与 benchmark 可执行文件 -----------------
add_executable(flat_hash_map_basic
    s_flat_hash_map/tests/flat_hash_map_basic.cpp
)

target_link_libraries(flat_hash_map_basic PRIVATE
    benchmark
    benchmark_main
)
