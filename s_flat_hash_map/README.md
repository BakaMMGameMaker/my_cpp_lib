# 高性能 flat hash map u32

## 未定义行为
- 不允许通过迭代器修改键值：`it->first` 改 key 会触发 UB。
- 默认构造或已失效迭代器的递增/解引用（如 `iterator it; ++it`）是 UB。
- 传入来自其他 map 的迭代器或范围调用 `erase`（`erase(it from other map)）是 UB。
补充1：容量改变后使用旧 it = UB（原因：元素在新表中的下标可能会发生改变）
补充2：u32 soa 插入和查找哨兵值是 UB（原因：key 为哨兵值的槽位会被视为没有 value）
补充3：在 size = 0 情况下进行 shrink_to_fit() 或者 rehash(0)，又在不 reserve 或者 rehash 就进行查找或者插入 = UB（原因：为了性能插入与查找不会检查容量是否大于 0）
补充4：erase + 不指向合法槽位的迭代器 = UB，因为 erase 里不会检查迭代器指向的槽位是否合法（虽然按照目前实现，每个 cluster 的开头元素都会在自己的 home 槽位，且处于 home 槽位的元素不会左移，所以 erase empty 不会有问题）
补充5：不提前预留足够的空间就调用 no rehash 系列接口，一旦没有任何空位，插入和查找都可能陷入死循环

## 预留与再哈希
- `reserve(n)` 以 size 语义工作：预留至少可容纳 `n` 个元素的空间。
- `rehash(n)` 以 capacity 语义工作：把桶数/容量调整到不小于 `n`。
- 任意 `erase` 操作都不会触发 rehash。

## 注意事项
insert 和 find hit 的高性能是以 erase 的低性能为代价的，本表只为 insert 与 find hit 服务，在 16 k 数量级场景下，平均探测长度为 1.5，在 256k 数量级下，平均探测长度为 2.5，所以没有引入 robin hood 和 SIMD 这种常数项极大的手段，也没有引入墓碑机制来产生更多的分支和 cache unfriendly 的代码。基于此，erase 的时候就是把右侧 cluster 整体左移一位，所以如果频繁 erase，将会导致哈希表的性能非常低下。