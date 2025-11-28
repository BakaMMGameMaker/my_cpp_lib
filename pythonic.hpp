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

template <LenAble Container> constexpr int len(const Container &c) noexcept { return static_cast<int>(c.size()); }

int len(...) = delete;
int len(const char *) = delete;
int len(char *) = delete;
int len(std::string_view) = delete;
inline int len(bool) = delete;