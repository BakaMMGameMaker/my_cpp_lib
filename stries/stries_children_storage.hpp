#pragma once
#include "salias.h"
#include "sutils.hpp"
#include <algorithm>
#include <array>
#include <memory>

// ForEachChildCallable 概念约束：需要能以 (SizeT, IndexType) 的形式被调用
template <typename Func, typename IndexType>
concept ForEachChildCallable = requires(Func &&func, UChar key, IndexType index) {
    { std::forward<Func>(func)(key, index) }; // 没有 forward 就只能保证 func 为左值时可以被调用
};

// Tries 树孩子存储数据结构概念约束
template <typename ChildrenStorageType>
concept TriesChildrenStorage =
    requires(ChildrenStorageType storage, const ChildrenStorageType const_storage, UChar key, UInt32 index) {
        { storage.get(key) } -> std::same_as<UInt32>;      // 以 UChar 为键，获取节点的索引
        { storage.set(key, index) } -> std::same_as<void>; // 设置以 UChar 为键的节点索引
        { storage.reset() } -> std::same_as<void>;         // 重置数据结构的状态
        { storage.empty() } -> std::convertible_to<bool>;  // 返回当前是否没有活跃的子节点
        { storage.size() } -> std::convertible_to<SizeT>;  // 返回当前活跃的子节点的个数
        {
            const_storage.for_each_child([](UChar, UInt32) {}) // 需要提供接口，使用传入的函数对象处理所有孩子节点
        } -> std::same_as<void>;
    };

// 仅用 array<Capacity> 来存储孩子节点，性能最高，但内存占用固定，开销较大
template <typename IndexType, SizeT Capacity> struct FixedChildren {
    static constexpr IndexType invalid_index = std::numeric_limits<IndexType>::max();

    std::array<IndexType, Capacity> children;
    SizeT active_count; // 活跃的孩子个数

    FixedChildren() : children(), active_count(0) { children.fill(invalid_index); }
    FixedChildren(const FixedChildren &) = delete;
    FixedChildren(FixedChildren &&) = default;
    FixedChildren &operator=(const FixedChildren &) = delete;
    FixedChildren &operator=(FixedChildren &&) noexcept = default;

    [[nodiscard]] IndexType get(UChar key) const noexcept { return children[CastSizeT(key)]; }

    void set(UChar key, IndexType index) noexcept {
        IndexType &ref = children[CastSizeT(key)];
        if (ref == invalid_index) {
            if (index == invalid_index) return;
            active_count++; // ref = invalid index and index != invalid index
        } else {
            if (index == invalid_index) --active_count; // ref != invalid index and index == invalid index
        }
        ref = index;
    }

    void reset() noexcept {
        children.fill(invalid_index);
        active_count = 0;
    }

    [[nodiscard]] SizeT size() const noexcept { return active_count; }
    [[nodiscard]] bool empty() const noexcept { return active_count == 0; }

    template <typename Func>
        requires ForEachChildCallable<Func, IndexType>
    void for_each_child(Func &&func) const {
        if (active_count == 0) return;
        for (SizeT key = 0; key < Capacity; ++key) {
            IndexType index = children[key];
            if (index != invalid_index) func(CastUChar(key), index);
        }
    }
};

// 子节点数量少时，用 array<Threshold> 来存储节点，可以在子节点数量少时避免 Capacity 次循环，性能较高，但内存开销更大
// 如果没有特殊要求，请使用 HybridDynamicChildren
template <typename IndexType, SizeT Capacity, SizeT Threshold, bool AllowShrinkToSparse = false>
struct [[deprecated("HybridFixedChildren is deprecated, use HybridDynamicChildren instead")]] HybridFixedChildren {
    static constexpr IndexType invalid_index = std::numeric_limits<IndexType>::max();

    struct Entry {
        UChar key;       // 0-255
        IndexType index; // 在节点池内的索引
    };

    std::array<Entry, Threshold> entries; // 还未到达阈值时
    bool using_dense;                     // 是否正在使用稠密数组

    std::array<IndexType, Capacity> dense; // 到达阈值后使用的稠密数组
    SizeT active_count;                    // 活跃的孩子个数

    HybridFixedChildren() : entries(), dense(), using_dense(false), active_count(0) {
        for (auto &e : entries) { e.index = invalid_index; }
        dense.fill(invalid_index);
    }
    HybridFixedChildren(const HybridFixedChildren &) = delete;
    HybridFixedChildren(HybridFixedChildren &&) = default;
    HybridFixedChildren &operator=(const HybridFixedChildren &) = delete;
    HybridFixedChildren &operator=(HybridFixedChildren &&) = default;

    [[nodiscard]] IndexType get(UChar key) const noexcept {
        if (using_dense) return dense[CastSizeT(key)]; // 如果正在使用稠密数组，直接返回对应槽位存储值

        // TODO：可能的 SIMD 优化
        for (SizeT i = 0; i < active_count; ++i) {
            if (entries[i].key == key) return entries[i].index;
        }
        return invalid_index;
    }

    void set(UChar key, IndexType index) noexcept {
        if (!using_dense) set_entry(key, index);
        else set_dense(key, index);
    }

    void reset() noexcept {
        active_count = 0;
        using_dense = false;
        dense.fill(invalid_index);
    }

    void set_entry(UChar key, IndexType index) noexcept {
        // 在有序 sparse 中找到 key 应该在的位置
        SizeT pos = 0;
        while (pos < active_count && entries[pos].key < key) pos++;

        // case 1 找到相同的 key
        if (pos < active_count && entries[pos].key == key) {
            if (index == invalid_index) {
                if (entries[pos].index == invalid_index) [[unlikely]]
                    return;
                // 删除：保持有序，左侧整体往左移动一格
                for (SizeT j = pos + 1; j < active_count; ++j) { entries[j - 1] = entries[j]; }
                active_count--;
            } else {
                entries[pos].index = index;
            }
            return;
        }

        // case 2 不存在 key 且试图删除
        if (index == invalid_index) return;

        // case 3 插入新的 key

        if (active_count < Threshold) {
            // 在 pos 位置插入，右侧整体往右移动一格
            for (SizeT j = active_count; j > pos; --j) { entries[j] = entries[j - 1]; }
            entries[pos].key = key;
            entries[pos].index = index;
            active_count++;
            return;
        }

        // 到达阈值，切换到稠密数组
        switch_to_dense();
        set_dense(key, index); // 不要遗漏临界数据
    }

    void set_dense(UChar key, IndexType index) noexcept {
        // 稠密数组路径
        IndexType &ref = dense[CastSizeT(key)];
        if (ref == invalid_index) {
            if (index == invalid_index) return;
            active_count++; // ref = invalid index and index != invalid index
        } else {
            if (index == invalid_index) active_count--; // ref != invalid index and index == invalid index
        }
        ref = index;

        if constexpr (AllowShrinkToSparse) {
            if (using_dense && active_count <= Threshold) switch_to_sparse();
        }
    }

    void switch_to_dense() noexcept {
        dense.fill(invalid_index);
        for (SizeT i = 0; i < active_count; ++i) {
            const Entry &entry = entries[i];
            dense[CastSizeT(entry.key)] = entry.index;
        }
        using_dense = true; // 切换到稠密数组模式
    }

    void switch_to_sparse() noexcept {
        SizeT count = 0;
        for (SizeT key = 0; key < Capacity; ++key) {
            IndexType index = dense[key];
            if (index == invalid_index) continue;
            entries[count].key = CastUChar(key);
            entries[count].index = index;
            count++;
        }
        active_count = count;
        using_dense = false;
    }

    [[nodiscard]] SizeT size() const noexcept { return active_count; }
    [[nodiscard]] bool empty() const noexcept { return active_count == 0; }

    template <typename Func>
        requires ForEachChildCallable<Func, IndexType>
    void for_each_child(Func &&func) const {
        if (active_count == 0) return;
        if (!using_dense) {
            for (SizeT i = 0; i < active_count; ++i) {
                const Entry &entry = entries[i];
                if (entry.index != invalid_index) [[likely]] // 防御
                    func(CastUChar(entry.key), entry.index);
            }
        } else {
            for (SizeT key = 0; key < Capacity; ++key) {
                IndexType index = dense[key];
                if (index != invalid_index) func(CastUChar(key), index);
            }
        }
    }
};

// 子节点数量少时使用 array<Threshold> 存储节点，仅在需要时扩展更多空间，内存友好，性能较高
template <typename IndexType, SizeT Capacity, SizeT Threshold, bool AllowShrinkToSparse = false>
struct HybridDynamicChildren {
    static constexpr IndexType invalid_index = std::numeric_limits<IndexType>::max();

    struct Entry {
        UChar key;
        IndexType index;
    };

    std::array<Entry, Threshold> entries;
    std::unique_ptr<IndexType[]> dense; // nullptr 表示尚未分配
    SizeT active_count;
    bool using_dense;

    HybridDynamicChildren() : entries(), dense(nullptr), active_count(0), using_dense(false) {
        for (auto &e : entries) { e.index = invalid_index; }
    }
    HybridDynamicChildren(const HybridDynamicChildren &) = delete;
    HybridDynamicChildren(HybridDynamicChildren &&) = default;
    HybridDynamicChildren &operator=(const HybridDynamicChildren &) = delete;
    HybridDynamicChildren &operator=(HybridDynamicChildren &&) = default;

    [[nodiscard]] IndexType get(UChar key) const noexcept {
        if (using_dense) {
            if (dense == nullptr) [[unlikely]]
                return invalid_index;
            return dense[CastSizeT(key)];
        }

        for (SizeT i = 0; i < active_count; ++i) {
            if (entries[i].key == key) return entries[i].index;
        }
        return invalid_index;
    }

    void set(UChar key, IndexType index) {
        if (!using_dense) set_entry(key, index);
        else set_dense(key, index);
    }

    void reset() noexcept {
        active_count = 0;
        using_dense = false;
        dense.reset(); // 把内存空间还给系统
    }

    void ensure_dense_allocated() {
        if (dense != nullptr) return;
        dense = std::make_unique_for_overwrite<IndexType[]>(Capacity);
        std::fill(dense.get(), dense.get() + Capacity, invalid_index);
    }

    void set_entry(UChar key, IndexType index) {
        SizeT pos = 0;
        while (pos < active_count && entries[pos].key < key) pos++;

        if (pos < active_count && entries[pos].key == key) {
            if (index == invalid_index) {
                if (entries[pos].index == invalid_index) [[unlikely]]
                    return;
                for (SizeT j = pos + 1; j < active_count; ++j) { entries[j - 1] = entries[j]; }
                active_count--;
            } else {
                entries[pos].index = index;
            }
            return;
        }

        if (index == invalid_index) return;

        if (active_count < Threshold) {
            for (SizeT j = active_count; j > pos; --j) { entries[j] = entries[j - 1]; }
            entries[pos].key = key;
            entries[pos].index = index;
            active_count++;
            return;
        }

        switch_to_dense();
        set_dense(key, index);
    }

    void set_dense(UChar key, IndexType index) {
        ensure_dense_allocated();
        IndexType &ref = dense[CastSizeT(key)];
        if (ref == invalid_index) {
            if (index == invalid_index) return;
            active_count++;
        } else {
            if (index == invalid_index) active_count--;
        }
        ref = index;

        if constexpr (AllowShrinkToSparse) {
            if (using_dense && active_count <= Threshold) switch_to_sparse();
        }
    }

    void switch_to_dense() {
        ensure_dense_allocated();
        for (SizeT i = 0; i < active_count; ++i) {
            const Entry &entry = entries[i];
            dense[CastSizeT(entry.key)] = entry.index;
        }
        using_dense = true;
    }

    void switch_to_sparse() {
        if (dense == nullptr) [[unlikely]] {
            using_dense = false;
            active_count = 0;
            return;
        }

        SizeT count = 0;
        for (SizeT key = 0; key < Capacity; ++key) {
            IndexType index = dense[key];
            if (index == invalid_index) continue;
            entries[count].key = CastUChar(key);
            entries[count].index = index;
            count++;
        }
        active_count = count;
        using_dense = false;
        dense.reset(); // 回收内存空间
    }

    [[nodiscard]] SizeT size() const noexcept { return active_count; }
    [[nodiscard]] bool empty() const noexcept { return active_count == 0; }

    template <typename Func>
        requires ForEachChildCallable<Func, IndexType>
    void for_each_child(Func &&func) const {
        if (active_count == 0) return;
        if (!using_dense) {
            for (SizeT i = 0; i < active_count; ++i) {
                const Entry &entry = entries[i];
                if (entry.index != invalid_index) [[likely]] // 防御
                    func(CastUChar(entry.key), entry.index);
            }
        } else {
            if (dense == nullptr) [[unlikely]]
                return;
            for (SizeT key = 0; key < Capacity; ++key) {
                IndexType index = dense[key];
                if (index != invalid_index) func(CastUChar(key), index);
            }
        }
    }
};