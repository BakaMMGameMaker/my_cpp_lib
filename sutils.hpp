#pragma once
#include "salias.h"
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>

template <typename T>
    requires std::convertible_to<T, std::size_t>
constexpr int CastInt(T value) noexcept {
    return static_cast<int>(value);
}

template <typename T>
    requires std::convertible_to<T, std::size_t>
constexpr int SafeCastInt(T value) {
    if constexpr (std::floating_point<T>) {
        // 浮点数精度检查
        if (value != static_cast<T>(static_cast<int>(value)))
            throw std::overflow_error("SafeCastInt: value cannot be represented exactly as int");
        // 浮点数范围检查
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
            throw std::overflow_error("SafeCastInt: value out of int range");
    } else {
        // 使用 std::in_range 安全地检查整数范围，可以自动处理有符号/无符号的边界情况
        if (!std::in_range<int>(value)) throw std::overflow_error("SafeCastInt: value out of int range");
    }
    return static_cast<int>(value);
}

template <typename T>
    requires std::convertible_to<T, std::size_t>
constexpr std::size_t CastSizeT(T value) noexcept {
    return static_cast<std::size_t>(value);
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
constexpr std::size_t SafeCastSizeT(T value) {
    if constexpr (std::floating_point<T>) {
        if (value < 0) throw std::invalid_argument("SafeCastSizeT: cannot cast negative value to std::size_t");
        if (value > static_cast<T>(std::numeric_limits<std::size_t>::max()))
            throw std::overflow_error("SafeCastSizeT: value too large for std::size_t");
    } else {
        if (!std::in_range<std::size_t>(value))
            throw std::invalid_argument("SafeCastSizeT: value out of std::size_t range (negative or too large)");
    }
    return static_cast<std::size_t>(value);
}

template <typename T>
concept LenAble = requires(const T &c) {
    { c.size() } -> std::convertible_to<std::size_t>;
};

template <LenAble Container> constexpr std::size_t len(const Container &c) noexcept { return c.size(); }

int len(...) = delete;
int len(const char *) = delete;
int len(char *) = delete;
int len(std::string_view) = delete;
inline int len(bool) = delete;

Int2D CreateInt2D(size_t rows, size_t cols, int value = 0);
UInt2D CreateUInt2D(size_t rows, size_t cols, size_t value = 0);
Bool2D CreateBool2D(size_t rows, size_t cols, bool value = false);