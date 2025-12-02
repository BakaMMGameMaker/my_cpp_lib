#pragma once

namespace sflat {

namespace detail {
// 定义控制字节 control_t = UInt8，有利于 SIMD，并避免花费过多时间比对 key
// - 0xFF(11111111) 表示 EMPTY
// - 0x80(10000000) 表示 DELETED
// - 最高位为 0 代表槽位被占据，即 FULL 范围为 0x00-0x7F (0-127)，冲突概率 1/128
// 虽然浪费了 10000001-11111110 但是为了极致的性能，这是必须放弃的编码空间
// 低 7 位放哈希高位混淆结果，避免全表线性扫描

} // namespace detail
} // namespace sflat