# s_flat_map_u32

基于平铺/Robin Hood 的 uint32_t 键哈希表，核心实现在 s_flat_map_u32.hpp。

## 文件
- s_flat_map_u32.hpp：flat_hash_map_u32<MappedType> 实现（单文件 header-only）。
- s_detail.h：hash 辅助、常量（最小容量 = 8，默认负载因子 = 0.75）。
- ../s_alias.h：UInt32/SizeT 类型别名。
- 相关：bench/、tests/ 提供性能与单测示例。

## 特性
- 开放定址 + 线性探测，容量保持 2 的幂；默认装载因子 0.75，可调。
- 专为 uint32_t 键，默认哈希 detail::FastUInt32Hash；空槽标记使用 UINT32_MAX。
- 可切换插入策略：default_policy、no_rehash、no_check_dup、no_return、fast 控制扩容/去重/返回值，可自定义。
- 迭代器为前向迭代器，begin()/end() 遍历已占用槽；支持 ==/swap。
- 提供 DEBUG 统计（探测长度、rehash 次数等）。

## 快速开始
```cpp
#include "s_flat_map_u32.hpp"
using mcl::flat_hash_map_u32;

int main() {
    flat_hash_map_u32<int> map(16);      // 预估 16 个元素
    map.emplace(1u, 42);
    map.try_emplace(2u, 7);
    map.insert_or_assign(1u, 99);

    if (map.contains(2u)) {
        auto &v = map.at(2u);
    }

    for (auto &[k, v] : map) { /* ... */ }
}
```
编译要求 C++20（使用 concepts）。包含路径需能找到同级 s_detail.h 与上级 s_alias.h。

## API 摘要
- 构造/资源：默认、std::initializer_list、拷贝/移动、swap、get_allocator。
- 容量：size/empty/capacity/growth_limit/load_factor/max_load_factor(lf)，reserve(n)、rehash(n)、shrink_to_fit()。
- 查找：find、find exist、contains、at、operator[]（需要 mapped_type 可默认构造）。
- 插入更新：insert、insert_or_assign、emplace、try_emplace、overwrite（假定键存在）、范围/初始化列表 insert。Policy 模板参数控制是否扩容/查重/返回值。
- 删除：erase(key)、erase(iterator)；erase_exist 针对必然存在的键。
- 迭代：begin/cbegin/end/cend，前向迭代器。

## 行为与注意事项
- 仅支持 uint32_t 键；UINT32_MAX 为空槽哨兵，不能作为有效 key。
- mapped_type 需可移动构造/析构；使用 operator[] 时还需可默认构造。
- find_exist、erase_exist、overwrite 假设键存在；若缺失会进入死循环，如果不确定键是否存在，转而调用其他安全接口。
- 扩容相关操作（insert/emplace/try_emplace/insert_or_assign 在达到 max_load_factor、显式reserve/rehash/shrink_to_fit）会重新分配，全部迭代器/引用都会失效。
- erase 采用后移补洞，会使被删元素及同簇后续元素的迭代器失效；不会自动收缩，需要手动 shrink_to_fit。
- 线程不安全；DEBUG 打开后会记录探测长度与 rehash 计数。
补充：为了减少 capacity == 0 的检查，map 创建后自动占用一些空间，可以在创建的时候显式传入 init_size = 0 来避免占用任何空间，但是后续插入元素的时候需要自行 reserve 或者 rehash，否则 UB
补充：如果在 size = 0 的情况下执行 shrink to fit 或者 rehash to 0，会归还所有空间，此时执行任何插入和查找都是 UB，记得 reserve

## 测试与基准
- 单测与基准位于 tests/、bench/，可参考根目录 CMakeLists.txt 配置。
