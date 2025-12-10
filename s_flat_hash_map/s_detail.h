#pragma once
#include "s_alias.h"

namespace mcl {
namespace detail {
// - 0x80(10000000) 表示 EMPTY
// - 0xFE(11111110) 表示 DELETED
// - 0xFF(11111111) 表示 SENTINEL
using control_t = UInt8; // 控制字节

// inline 允许跨多翻译单元重复定义，适用于 header-only 常量
inline constexpr control_t k_empty = static_cast<control_t>(-128); // 0x80 EMPTY
// inline constexpr control_t k_deleted = static_cast<control_t>(-2); // 0xFE DELETED [[deprecated]]
inline constexpr control_t k_sentinel = static_cast<control_t>(-1); // 0xFF SENTINAL
inline constexpr UInt32 k_min_capacity = 8;                         // 逻辑允许最大元素数量 = capacity * max load factor
inline constexpr float k_default_max_load_factor = 0.75f;
inline constexpr int k_group_width = 16; // 一组 16 个 control_t

// 数值是否为 2 的次幂
inline constexpr bool is_power_of_two(UInt32 x) noexcept { return x && ((x & (x - 1)) == 0); }

inline constexpr UInt32 next_power_of_two(UInt32 x) noexcept {
    --x;
    x |= x >> 1; // 最高位和第二高位都为 1
    x |= x >> 2; // 前四高位都为 1
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16; // 确保 32 位都为 1
    return x + 1;
}

struct FastUInt32Hash {
    [[nodiscard]] UInt32 operator()(UInt32 x) const noexcept {
        constexpr UInt32 k_mul = 0x9E3779B1u;
        return static_cast<UInt32>(x * k_mul);
    }
};

// 7-bit hash 指纹
inline constexpr control_t short_hash(UInt32 h) noexcept {
    // note: h >> 7 是个可调参数，只要 >=7，不要取低位即可
    UInt8 v = static_cast<UInt8>((h >> 8) & 0x7Fu); // 0x7F = 0111 1111
    return static_cast<control_t>(v);
}

template <typename KeyType>
concept HashKey = std::copy_constructible<KeyType> && std::equality_comparable<KeyType> &&
                  std::is_nothrow_move_constructible_v<KeyType>;

template <typename ValueType>
concept HashValue = std::default_initializable<ValueType> && std::move_constructible<ValueType> &&
                    std::destructible<ValueType> && std::is_nothrow_move_constructible_v<ValueType>;

// Hasher 概念约束
template <typename HasherType, typename KeyType>
concept HashFor =
    requires(const HasherType &hasher, const KeyType &key) {
        { hasher(key) } noexcept -> std::convertible_to<UInt32>; // hasher 可调用且无抛，必须返回 size_t
    } && std::is_nothrow_move_constructible_v<HasherType>        // 能移动构造且无抛
    && std::is_nothrow_default_constructible_v<HasherType>;      // 能默认构造且无抛

// Equal 概念约束
template <typename KeyEqualType, typename KeyType>
concept EqualFor = requires(const KeyEqualType &e, const KeyType &a, const KeyType &b) {
    { e(a, b) } noexcept -> std::same_as<bool>; // 必须返回 bool 且无抛
} && std::is_nothrow_move_constructible_v<KeyEqualType> && std::is_nothrow_default_constructible_v<KeyEqualType>;

// u32 与 u64 不存 full hash，现场算
template <typename Key, typename Hash>
inline constexpr bool k_store_hash =
    !(std::is_integral_v<Key> && (sizeof(Key) == 4 || sizeof(Key) == 8) && std::is_same_v<Hash, std::hash<Key>>);

} // namespace detail
} // namespace mcl