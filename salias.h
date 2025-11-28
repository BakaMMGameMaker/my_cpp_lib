#pragma once
#include <cstddef>
#include <vector>

template <typename T> using Vec1D = std::vector<T>;
template <typename T> using Vec2D = std::vector<std::vector<T>>;

using Int1D = Vec1D<int>;
using Int2D = Vec2D<int>;
using UInt1D = Vec1D<size_t>;
using UInt2D = Vec2D<size_t>;
using Bool1D = Vec1D<char>;
using Bool2D = Vec2D<char>;

template <typename T> Vec2D<T> CreateVec2D(size_t rows, size_t cols, T value = T{}) {
    return Vec2D<T>(rows, Vec1D<T>(cols, value));
}