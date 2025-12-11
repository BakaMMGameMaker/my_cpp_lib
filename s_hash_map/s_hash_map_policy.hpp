#pragma once

#include <concepts>

namespace mcl {
template <typename P>
concept InsertPolicy = requires {
    { P::return_value } -> std::convertible_to<bool>; // 是否返回值
    { P::rehash } -> std::convertible_to<bool>;       // 是否触发扩容
    { P::check_dup } -> std::convertible_to<bool>;    // 是否检查重复键
};

struct default_policy {
    static constexpr bool return_value = true;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = true;
};

// 不返回值
struct no_return {
    static constexpr bool return_value = false;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = true;
};

// 不扩容
struct no_rehash {
    static constexpr bool return_value = true;
    static constexpr bool rehash = false;
    static constexpr bool check_dup = true;
};

// 不查重
struct no_check_dup {
    static constexpr bool return_value = true;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = false;
};

struct insert_range {
    static constexpr bool return_value = false;
    static constexpr bool rehash = false;
    static constexpr bool check_dup = true;
};

// 仅插入元素
struct fast {
    static constexpr bool return_value = false;
    static constexpr bool rehash = false;
    static constexpr bool check_dup = false;
};
} // namespace mcl