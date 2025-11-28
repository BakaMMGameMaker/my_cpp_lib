#pragma once
#include <concepts>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

template <typename T>
concept LenAble = requires(const T &c) {
    { c.size() } -> std::convertible_to<std::size_t>;
};

/**
 * @brief 长度函数，返回 std::size_t
 * @tparam Container 含有 .size() 方法且返回 std::size_t 的类型
 * @param  c         传入的容器
 * @return std::size_t 容器元素个数
 */
template <LenAble Container> constexpr std::size_t len(const Container &c) noexcept { return c.size(); }

int len(...) = delete;
int len(const char *) = delete;
int len(char *) = delete;
int len(std::string_view) = delete;
inline int len(bool) = delete;

/**
 * @brief
 *
 * 把一个数值转化为 int 并返回
 *
 * @tparam T       支持转为 std::size_t 的数值类型
 * @param  value   要强制类型转换的值
 * @return int     强制类型转换后的结果
 */
template <typename T>
    requires std::convertible_to<T, std::size_t>
constexpr int Int(T value) noexcept {
    return static_cast<int>(value);
}