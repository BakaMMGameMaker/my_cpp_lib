#pragma once
#include <concepts>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief 把一个数值转化为 int 并返回
 * @tparam T 支持转为 std::size_t 的数值类型
 * @param  value 要强制类型转换的值
 * @return int 强制类型转换后的结果
 */
template <typename T>
    requires std::convertible_to<T, std::size_t>
constexpr int CastInt(T value) noexcept {
    return static_cast<int>(value);
}

/**
 * @brief 把一个数值转化为 int 并返回
 * @tparam T 支持转为 std::size_t 的数值类型
 * @param  value 要强制类型转换的值，值太大或为负数时抛出
 * @return int 强制类型转换后的结果
 */
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

/**
 * @brief 把一个数值转化为 std::size_t 并返回
 * @tparam T 支持转为 std::size_t 的数值类型
 * @param  value 要强制类型转换的值
 * @return std::size_t 强制类型转换后的结果
 */
template <typename T>
    requires std::convertible_to<T, std::size_t>
constexpr std::size_t CastSizeT(T value) noexcept {
    return static_cast<std::size_t>(value);
}

/**
 * @brief 把一个数值安全地转化为 std::size_t 并返回
 * @tparam T 支持转为 std::size_t 的数值类型
 * @param  value 要强制类型转换的值，若为负数，抛出
 * @return std::size_t 强制类型转换后的结果
 */
template <typename T>
    requires std::integral<T> || std::floating_point<T>
constexpr std::size_t SafeCastSizeT(T value) {
    if constexpr (std::floating_point<T>) {
        if (value < 0) throw std::invalid_argument("SafeCastSizeT: cannot cast negative value to std::size_t");
    } else {
        if (!std::in_range<std::size_t>(value))
            throw std::invalid_argument("SafeCastSizeT: value out of std::size_t range (negative or too large)");
    }
    return static_cast<std::size_t>(value);
}

/**
 * @brief 基于 std::vector<std::vector<T>> 实现的二维矩阵
 * @tparam T 矩阵存储的元素类型
 */
template <typename T> class Matrix2D {
private:
    std::vector<std::vector<T>> data_;

    /**
     * @brief 私有构造函数，用户通过 CreateSafe/CreateUnsafe 创建
     */
    Matrix2D(std::size_t rows, std::size_t cols, const T &default_value = T{})
        : data_(rows, std::vector<T>(cols, default_value)) {}

public:
    /**
     * @brief 创建矩阵，检查参数合法性
     * @param rows 行数 (必须 > 0)
     * @param cols 列数 (必须 > 0)
     * @param default_value 默认值
     * @throws std::invalid_argument 如果行列为0
     */
    [[nodiscard]] static Matrix2D SafeCreate(std::size_t rows, std::size_t cols, const T &default_value = T{}) {
        if (rows == 0 || cols == 0)
            throw std::invalid_argument("Matrix2D::SafeCreate: Dimensions must be positive (rows > 0 and cols > 0)");
        return Matrix2D(rows, cols, default_value);
    }

    /**
     * @brief 创建矩阵，不做检查
     * @note  注意：std::vector 仍然会处理内存分配失败的情况(抛出 bad_alloc)，
     */
    [[nodiscard]] static Matrix2D Create(std::size_t rows, std::size_t cols, const T &default_value = T{}) {
        return Matrix2D(rows, cols, default_value);
    }

    constexpr std::size_t rows() const noexcept { return data_.size(); }
    constexpr std::size_t cols() const noexcept { return data_.empty() ? 0 : data_[0].size(); }
    constexpr std::size_t size() const noexcept { return rows() * cols(); }

    const std::vector<T> &operator[](std::size_t row) const { return data_[row]; }
    std::vector<T> &operator[](std::size_t row) { return data_[row]; }

    /**
     * @brief 获取指定位置元素的只读引用 (带边界检查)
     * @param row 行索引
     * @param col 列索引
     * @return const T& 元素的引用
     * @throws std::out_of_range 如果行或列索引越界
     */
    const T &SafeAt(std::size_t row, std::size_t col) const {
        if (row >= rows()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: row index out of range");
        if (col >= data_[row].size()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: col index out of range");
        return data_[row][col];
    }

    /**
     * @brief 获取指定位置元素的可变引用 (带边界检查)
     * @param row 行索引
     * @param col 列索引
     * @return T& 元素的引用
     * @throws std::out_of_range 如果行或列索引越界
     */
    [[nodiscard]] T &SafeAt(std::size_t row, std::size_t col) {
        if (row >= rows()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: row index out of range");
        if (col >= data_[row].size()) [[unlikely]]
            throw std::out_of_range("Matrix2D::SafeAt: col index out of range");
        return data_[row][col];
    }

    /**
     * @brief 获取指定位置元素的只读引用 (无边界检查)
     * @param row 行索引
     * @param col 列索引
     * @return const T& 元素的引用
     */
    [[nodiscard]] const T &At(std::size_t row, std::size_t col) const noexcept { return data_[row][col]; }

    /**
     * @brief 获取指定位置元素的可变引用 (无边界检查)
     * @param row 行索引
     * @param col 列索引
     * @return T& 元素的引用
     */
    [[nodiscard]] T &At(std::size_t row, std::size_t col) noexcept { return data_[row][col]; }
};

template <typename T>
concept LenAble = requires(const T &c) {
    { c.size() } -> std::convertible_to<std::size_t>;
};

/**
 * @brief 长度函数，返回 std::size_t
 * @tparam Container 含有 .size() 方法且返回 std::size_t 的类型
 * @param c 传入的容器
 * @return std::size_t 容器元素个数
 */
template <LenAble Container> constexpr std::size_t len(const Container &c) noexcept { return c.size(); }

int len(...) = delete;
int len(const char *) = delete;
int len(char *) = delete;
int len(std::string_view) = delete;
inline int len(bool) = delete;