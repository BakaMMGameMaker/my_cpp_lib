#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

template <typename T> using Vec1D = std::vector<T>;
template <typename T> using Vec2D = std::vector<std::vector<T>>;

using Int8 = std::int8_t;
using Int16 = std::int16_t;
using Int32 = std::int32_t;
using Int64 = std::int64_t;
using UChar = unsigned char;
using UInt8 = std::uint8_t;
using UInt16 = std::uint16_t;
using UInt32 = std::uint32_t;
using UInt64 = std::uint64_t;
using SizeT = std::size_t;
using Int1D = Vec1D<int>;
using Int2D = Vec2D<int>;
using UInt1D = Vec1D<size_t>;
using UInt2D = Vec2D<size_t>;
using Bool1D = Vec1D<char>;
using Bool2D = Vec2D<char>;

constexpr SizeT SizeMax = std::numeric_limits<SizeT>::max();
