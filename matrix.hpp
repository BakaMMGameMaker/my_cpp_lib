#pragma once
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
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
    [[nodiscard]] std::vector<T> Col(std::size_t col) const {
        std::vector<T> result;
        result.reserve(rows());
        for (const auto &row_vec : data_) result.push_back(row_vec[col]);
        return result;
    }
    [[nodiscard]] std::vector<T> SafeCol(std::size_t col) const {
        if (col >= cols()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeCol: col index out of range");
        return Col(col);
    }
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
