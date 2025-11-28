#pragma once
#include <vector>

template <typename T> using Vec1D = std::vector<T>;
template <typename T> using Vec2D = std::vector<std::vector<T>>;

using Int1D = Vec1D<int>;
using Int2D = Vec2D<int>;
using Bool1D = Vec1D<char>;
using Bool2D = Vec2D<char>;

template <typename T> auto CreateVec2D(int rows, int cols, T value) { return Vec2D<T>(rows, Vec1D<T>(cols, value)); }