# 高性能游戏向 flat hash map

## 未定义行为
- 不允许通过迭代器修改键值：`it->first` 改 key 会触发 UB。
- 默认构造或已失效迭代器的递增/解引用（如 `iterator it; ++it`）是 UB。
- 传入来自其他 map 的迭代器或范围调用 `erase`（`erase(it from other map)）是 UB。
补充1：容量改变后使用旧 it = UB
补充2：u32 soa 插入和查找哨兵值是 UB
## 预留与再哈希
- `reserve(n)` 以 size 语义工作：预留至少可容纳 `n` 个元素的空间。
- `rehash(n)` 以 capacity 语义工作：把桶数/容量调整到不小于 `n`。
- 任意 `erase` 操作都不会触发 rehash。
