#pragma once
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>


template <typename T> class Matrix2D {
    static_assert(!std::is_same_v<T, bool>, "Matrix2D does not support bool, use char instead");

private:
    std::vector<std::vector<T>> data_;

    Matrix2D(std::size_t rows, std::size_t cols, const T &default_value = T{})
        : data_(rows, std::vector<T>(cols, default_value)) {}

public:
    Matrix2D(const Matrix2D &) = default;
    Matrix2D(Matrix2D &&) noexcept = default;
    Matrix2D &operator=(const Matrix2D &) = default;
    Matrix2D &operator=(Matrix2D &&) noexcept = default;

    [[nodiscard]] static Matrix2D SafeCreate(std::size_t rows, std::size_t cols, const T &default_value = T{}) {
        if (rows == 0 || cols == 0)
            throw std::invalid_argument("Matrix2D::SafeCreate: Dimensions must be positive (rows > 0 and cols > 0)");
        // 检查乘法溢出
        if (rows > std::numeric_limits<std::size_t>::max() / cols)
            throw std::overflow_error("Matrix2D::SafeCreate: Total size (rows * cols) exceeds std::size_t limit");
        return Matrix2D(rows, cols, default_value);
    }
    [[nodiscard]] static Matrix2D Create(std::size_t rows, std::size_t cols, const T &default_value = T{}) {
        return Matrix2D(rows, cols, default_value);
    }

    constexpr std::size_t rows() const noexcept { return data_.size(); }
    constexpr std::size_t cols() const noexcept { return data_.empty() ? 0 : data_[0].size(); }
    constexpr std::size_t size() const noexcept { return rows() * cols(); }

    // 使用 span 防止用户更改 vector 大小
    std::span<const T> operator[](std::size_t row) const noexcept { return data_[row]; }
    std::span<T> operator[](std::size_t row) noexcept { return data_[row]; }
    [[nodiscard]] std::span<const T> Row(std::size_t row) const noexcept { return data_[row]; }
    [[nodiscard]] std::span<T> Row(std::size_t row) noexcept { return data_[row]; }
    [[nodiscard]] std::span<const T> SafeRow(std::size_t row) const {
        if (row >= rows()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeRow: row index out of range");
        return data_[row];
    }
    [[nodiscard]] std::span<T> SafeRow(std::size_t row) {
        if (row >= rows()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeRow: row index out of range");
        return data_[row];
    }
    [[nodiscard]] std::vector<T> Col(std::size_t col) const {
        std::vector<T> result;
        result.reserve(rows());
        for (const auto &row_vec : data_) { result.push_back(row_vec[col]); }
        return result;
    }
    [[nodiscard]] std::vector<T> SafeCol(std::size_t col) const {
        if (col >= cols()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeCol: col index out of range");
        return Col(col);
    }
    [[nodiscard]] const T &SafeAt(std::size_t row, std::size_t col) const {
        if (row >= rows()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: row index out of range");
        if (col >= data_[row].size()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: col index out of range");
        return data_[row][col];
    }
    [[nodiscard]] T &SafeAt(std::size_t row, std::size_t col) {
        if (row >= rows()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: row index out of range");
        if (col >= data_[row].size()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: col index out of range");
        return data_[row][col];
    }
    // 返回 T 防止悬空引用，用户传入临时对象时，返回 fallback 会导致悬空引用
    [[nodiscard]] T SafeAt(std::size_t row, std::size_t col, const T &fallback) const noexcept {
        if (row >= rows() || col >= cols()) [[unlikely]] { return fallback; }
        return data_[row][col];
    }
    [[nodiscard]] const T &At(std::size_t row, std::size_t col) const noexcept { return data_[row][col]; }
    [[nodiscard]] T &At(std::size_t row, std::size_t col) noexcept { return data_[row][col]; }
};

template <typename T> class FlatMatrix2D {
    static_assert(!std::is_same_v<T, bool>, "FlatMatrix2D does not support bool, use char instead");

private:
    std::vector<T> data_;
    std::size_t rows_{0};
    std::size_t cols_{0};

    FlatMatrix2D(std::size_t rows, std::size_t cols, const T &default_value)
        : data_(rows * cols, default_value), rows_(rows), cols_(cols) {}

    [[nodiscard]] constexpr std::size_t index(std::size_t row, std::size_t col) const noexcept {
        return row * cols_ + col;
    }

public:
    FlatMatrix2D(const FlatMatrix2D &) = default;
    FlatMatrix2D(FlatMatrix2D &&) noexcept = default;
    FlatMatrix2D &operator=(const FlatMatrix2D &) = default;
    FlatMatrix2D &operator=(FlatMatrix2D &&) noexcept = default;

    [[nodiscard]] static FlatMatrix2D SafeCreate(std::size_t rows, std::size_t cols, const T &default_value = T{}) {
        if (rows == 0 || cols == 0)
            throw std::invalid_argument("FlatMatrix2D::SafeCreate: Dimensions must be positive");
        if (rows > std::numeric_limits<std::size_t>::max() / cols)
            throw std::overflow_error("FlatMatrix2D::SafeCreate: Total size exceeds std::size_t limit");
        return FlatMatrix2D(rows, cols, default_value);
    }
    [[nodiscard]] static FlatMatrix2D Create(std::size_t rows, std::size_t cols, const T &default_value = T{}) {
        return FlatMatrix2D(rows, cols, default_value);
    }

    [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return data_.size(); }

    // 便于与 C API 交互
    [[nodiscard]] const T *data() const noexcept { return data_.data(); }
    [[nodiscard]] T *data() noexcept { return data_.data(); }

    [[nodiscard]] std::span<const T> operator[](std::size_t row) const noexcept {
        return std::span<const T>(data_.data() + row * cols_, cols_);
    }
    [[nodiscard]] std::span<T> operator[](std::size_t row) noexcept {
        return std::span<T>(data_.data() + row * cols_, cols_);
    }

    [[nodiscard]] std::span<const T> Row(std::size_t row) const noexcept { return (*this)[row]; }
    [[nodiscard]] std::span<T> Row(std::size_t row) noexcept { return (*this)[row]; }
    [[nodiscard]] std::span<const T> SafeRow(std::size_t row) const {
        if (row >= rows_) [[unlikely]]
            throw std::out_of_range("FlatMatrix2D::SafeRow: row index out of range");
        return (*this)[row];
    }
    [[nodiscard]] std::span<T> SafeRow(std::size_t row) {
        if (row >= rows_) [[unlikely]]
            throw std::out_of_range("FlatMatrix2D::SafeRow: row index out of range");
        return (*this)[row];
    }
    [[nodiscard]] std::vector<T> Col(std::size_t col) const {
        std::vector<T> result;
        result.reserve(rows_);
        for (std::size_t r = 0; r < rows_; ++r) { result.push_back(data_[index(r, col)]); }
        return result;
    }
    [[nodiscard]] std::vector<T> SafeCol(std::size_t col) const {
        if (col >= cols_) [[unlikely]]
            throw std::out_of_range("FlatMatrix2D::SafeCol: col index out of range");
        return Col(col);
    }

    [[nodiscard]] const T &SafeAt(std::size_t row, std::size_t col) const {
        if (row >= rows_ || col >= cols_) [[unlikely]]
            throw std::out_of_range("FlatMatrix2D::SafeAt: index out of range");
        return data_[index(row, col)];
    }
    [[nodiscard]] T &SafeAt(std::size_t row, std::size_t col) {
        if (row >= rows_ || col >= cols_) [[unlikely]]
            throw std::out_of_range("FlatMatrix2D::SafeAt: index out of range");
        return data_[index(row, col)];
    }
    [[nodiscard]] T SafeAt(std::size_t row, std::size_t col, const T &fallback) const noexcept {
        if (row >= rows_ || col >= cols_) [[unlikely]]
            return fallback;
        return data_[index(row, col)];
    }
    [[nodiscard]] const T &At(std::size_t row, std::size_t col) const noexcept { return data_[index(row, col)]; }
    [[nodiscard]] T &At(std::size_t row, std::size_t col) noexcept { return data_[index(row, col)]; }

    void Resize(std::size_t new_rows, std::size_t new_cols, const T &default_value = T{}) {
        if (new_rows > std::numeric_limits<std::size_t>::max() / new_cols)
            throw std::overflow_error("FlatMatrix2D::Resize: Total size exceeds limit");
        data_.assign(new_rows * new_cols, default_value);
        rows_ = new_rows;
        cols_ = new_cols;
    }
};