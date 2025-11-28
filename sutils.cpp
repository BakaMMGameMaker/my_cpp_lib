#include "sutils.hpp"
#include "salias.h"
#include <cstddef>

Int2D CreateInt2D(size_t rows, size_t cols, int value) { return Int2D(rows, Int1D(cols, value)); }
UInt2D CreateUInt2D(size_t rows, size_t cols, size_t value) { return UInt2D(rows, UInt1D(cols, value)); }
Bool2D CreateBool2D(size_t rows, size_t cols, bool value) { return Bool2D(rows, Bool1D(cols, value ? 1 : 0)); }