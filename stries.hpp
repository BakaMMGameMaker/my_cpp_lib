#pragma once
#include "salias.h"
#include "sutils.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

// ForEachChildCallable 概念约束：需要能以 (SizeT, IndexType) 的形式被调用
template <typename Func, typename IndexType>
concept ForEachChildCallable = requires(Func &&func, UChar key, IndexType index) {
    { std::forward<Func>(func)(key, index) }; // 没有 forward 就只能保证 func 为左值时可以被调用
};

// ChildrenStorageType 概念约束：
// - get(SizeT) -> IndexType
// - set(SizeT, IndexType) -> void
// - empty() -> bool
// - size() -> SizeT
// - for_each_child(Func) -> void
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

// 仅用 array<Capacity> 来存储孩子节点，性能较高，但内存开销较大
template <typename IndexType, SizeT Capacity> struct FixedChildren {
    static constexpr IndexType invalid_index = std::numeric_limits<IndexType>::max();

    std::array<IndexType, Capacity> children;
    SizeT active_count; // 活跃的孩子个数

    FixedChildren() : children(), active_count(0) { children.fill(invalid_index); }

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
template <typename IndexType, SizeT Capacity, SizeT Threshold, bool AllowShrinkToSparse = false>
struct HybridFixedChildren {
    static constexpr IndexType invalid_index = std::numeric_limits<IndexType>::max();

    struct Entry {
        UChar key;       // 0-255
        IndexType index; // 在节点池内的索引
    };

    std::array<Entry, Threshold> entries; // 还未到达阈值时
    bool using_dense;                     // 是否正在使用稠密数组

    std::array<IndexType, Capacity> dense; // 到达阈值后使用的稠密数组
    SizeT active_count;                    // 活跃的孩子个数

    HybridFixedChildren() : entries(), dense(), using_dense(false), active_count(0) {}

    [[nodiscard]] IndexType get(UChar key) const noexcept {
        if (using_dense) return dense[CastSizeT(key)]; // 如果正在使用稠密数组，直接返回对应槽位存储值

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
                if (entries[i].index == invalid_index) [[unlikely]] // 防御
                    continue;
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
struct HybridHeapChildren {
    static constexpr IndexType invalid_index = std::numeric_limits<IndexType>::max();

    struct Entry {
        UChar key;
        IndexType index;
    };

    std::array<Entry, Threshold> entries;
    std::unique_ptr<IndexType[]> dense; // nullptr 表示尚未分配
    bool using_dense;
    SizeT active_count;

    HybridHeapChildren() : entries(), dense(nullptr), using_dense(false), active_count(0) {}

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
        dense = std::make_unique<IndexType[]>(Capacity);
        for (SizeT i = 0; i < Capacity; ++i) dense[i] = invalid_index;
    }

    void set_entry(UChar key, IndexType index) {
        SizeT pos = 0;
        while (pos < active_count && entries[pos].key < key) pos++;

        if (pos < active_count && entries[pos].key == key) {
            if (index == invalid_index) {
                if (entries.index == invalid_index) [[unlikely]]
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
                if (entry.index == invalid_index) [[unlikely]]
                    continue;
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

// FreeList 的结构，用于开关死节点复用特性
template <typename IndexType, bool Enable> struct TriesFreeList;

template <typename IndexType> struct TriesFreeList<IndexType, false> {
    explicit TriesFreeList(std::pmr::memory_resource *) noexcept {}

    [[nodiscard]] bool empty() const noexcept { return true; }
    void push(IndexType) noexcept {}
    [[nodiscard]] IndexType pop() noexcept { return {}; }
};

template <typename IndexType> struct TriesFreeList<IndexType, true> {
    std::pmr::vector<IndexType> indices;
    explicit TriesFreeList(std::pmr::memory_resource *resource) : indices(resource) {}

    [[nodiscard]] bool empty() const noexcept { return indices.empty(); }
    void push(IndexType index) { indices.push_back(index); }
    [[nodiscard]] IndexType pop() {
        IndexType index = indices.back();
        indices.pop_back();
        return index;
    }
};

// CRTP base
template <typename Derived> class TriesBase {
protected:
    TriesBase() = default;
    ~TriesBase() = default;

    TriesBase(const TriesBase &) = delete;
    TriesBase &operator=(const TriesBase &) = delete;
    TriesBase(TriesBase &&) = delete;
    TriesBase &operator=(TriesBase &&) = delete;

    [[nodiscard]] Derived &derived() noexcept { return static_cast<Derived &>(*this); }
    [[nodiscard]] const Derived &derived() const noexcept { return static_cast<const Derived &>(*this); }

    mutable std::shared_mutex shared_mutex_;

public:
    // 添加新的单词
    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex_);
        derived().push_impl(word);
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex_);
        return derived().erase_impl(word);
    }

    // 查找是否包含给定单词
    [[nodiscard]] bool contains(std::string_view word) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex_);
        return derived().contains_impl(word);
    }

    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex_);
        return derived().contains_starts_with_impl(prefix);
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, SizeT limit = SizeMax) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex_);
        return derived().prefix_search_impl(prefix, limit);
    }
};

// 原生的链式结构 Tries
class STries : public TriesBase<STries> {
    friend class TriesBase<STries>;

    struct Node {
        std::map<char, std::unique_ptr<Node>> children;
        bool is_end = false;
    };

    std::unique_ptr<Node> root = std::make_unique<Node>();

    // 深度优先搜索收集结果
    void dfs(const Node *node, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;
        if (node->is_end) {
            result.push_back(current_word);
            if (result.size() >= limit) return;
        }

        for (auto &[ch, child] : node->children) {
            current_word.push_back(ch);
            dfs(child.get(), current_word, result, limit);
            current_word.pop_back();
            if (result.size() >= limit) return;
        }
    }

    // 定位前缀对应的节点
    [[nodiscard]] const Node *find_node(std::string_view prefix) const noexcept {
        const Node *current_node = root.get();
        for (char ch : prefix) {
            auto it = current_node->children.find(ch);
            if (it == current_node->children.end()) return nullptr;
            current_node = it->second.get();
        }
        return current_node;
    }

    void push_impl(std::string_view word) {
        Node *current_node = root.get();
        for (char ch : word) {
            auto it = current_node->children.find(ch);
            if (it == current_node->children.end()) {
                auto insert_result = current_node->children.try_emplace(ch, std::make_unique<Node>());
                it = insert_result.first;
            }
            current_node = it->second.get();
        }
        current_node->is_end = true;
    }

    [[nodiscard]] bool contains_impl(std::string_view word) const noexcept {
        const Node *node = find_node(word);
        return node != nullptr && node->is_end;
    }

    bool erase_impl(std::string_view word) {
        Node *current_node = root.get();
        std::vector<std::pair<Node *, char>> path; // 记录路径
        path.reserve(word.size());

        for (char ch : word) {
            auto it = current_node->children.find(ch);
            if (it == current_node->children.end()) return false; // word 不存在
            path.emplace_back(current_node, ch);
            current_node = it->second.get();
        }

        if (!current_node->is_end) return false; // 当前单词仅为前缀，非完整单词
        current_node->is_end = false;            // 把单词末尾标记为非 end
        if (!current_node->children.empty()) return true;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            Node *parent = it->first;
            char edge = it->second;
            auto child_it = parent->children.find(edge);
            if (child_it == parent->children.end()) [[unlikely]]
                break;

            const Node *child = child_it->second.get();
            if (child->is_end || !child->children.empty()) break; // 不作为某单词的结尾，也不属于任何单词的前缀部分
            parent->children.erase(child_it);
        }
        return true;
    }

    [[nodiscard]] bool contains_starts_with_impl(std::string_view prefix) const noexcept { return find_node(prefix); }

    [[nodiscard]] std::vector<std::string> prefix_search_impl(std::string_view prefix, SizeT limit = SizeMax) const {
        const Node *start_node = find_node(prefix);
        if (start_node == nullptr) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node, current_word, result, limit);
        return result;
    }

public:
    STries() = default;
};

// 使用对象池存储节点
class SFlatTries : public TriesBase<SFlatTries> {
    friend class TriesBase<SFlatTries>;

    struct Node {
        std::map<char, UInt32> children;
        bool is_end = false;
    };

    std::vector<Node> node_pool; // index 是在池中的下标
    static constexpr UInt32 invalid_index = std::numeric_limits<UInt32>::max();

    UInt32 create_node() {
        node_pool.emplace_back();
        return static_cast<UInt32>(node_pool.size() - 1);
    }

    void dfs(UInt32 node_index, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;
        const Node &current_node = node_pool[node_index];
        if (current_node.is_end) {
            result.push_back(current_word);
            if (result.size() >= limit) return;
        }

        for (const auto &[ch, child_node_index] : current_node.children) {
            current_word.push_back(ch);
            dfs(child_node_index, current_word, result, limit);
            current_word.pop_back();
            if (result.size() >= limit) return;
        }
    }

    [[nodiscard]] UInt32 find_node_index(std::string_view prefix) const noexcept {
        UInt32 current_node_index = 0; // root
        for (char ch : prefix) {
            const Node &current_node = node_pool[current_node_index];
            auto it = current_node.children.find(ch);
            if (it == current_node.children.end()) return invalid_index;
            current_node_index = it->second;
        }
        return current_node_index;
    }

    void push_impl(std::string_view word) {
        UInt32 current_node_index = 0; // root
        for (char ch : word) {
            // 这里不应该使用引用，即便是单线程，create node 也会导致悬空引用
            auto it = node_pool[current_node_index].children.find(ch);
            if (it == node_pool[current_node_index].children.end()) {
                UInt32 child_node_index = create_node(); // 此处可能导致 reallocate
                node_pool[current_node_index].children.try_emplace(ch, child_node_index);
                current_node_index = child_node_index;
            } else {
                current_node_index = it->second;
            }
        }
        node_pool[current_node_index].is_end = true;
    }

    [[nodiscard]] bool contains_impl(std::string_view word) const noexcept {
        UInt32 node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    bool erase_impl(std::string_view word) {
        UInt32 current_node_index = 0; // root
        std::vector<std::pair<UInt32, char>> path;
        path.reserve(word.size());

        for (char ch : word) {
            Node &current_node = node_pool[current_node_index];
            auto it = current_node.children.find(ch);
            if (it == current_node.children.end()) return false;
            path.emplace_back(current_node_index, ch);
            current_node_index = it->second;
        }

        Node &terminal_node = node_pool[current_node_index];
        if (!terminal_node.is_end) return false;
        terminal_node.is_end = false;
        if (!terminal_node.children.empty()) return true;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            UInt32 parent_index = it->first;
            char edge = it->second;
            Node &parent_node = node_pool[parent_index];
            auto child_it = parent_node.children.find(edge);
            if (child_it == parent_node.children.end()) [[unlikely]]
                break;

            UInt32 child_node_index = child_it->second;
            const Node &child_node = node_pool[child_node_index];
            if (child_node.is_end || !child_node.children.empty()) break;
            parent_node.children.erase(child_it);
        }
        return true;
    }

    [[nodiscard]] bool contains_starts_with_impl(std::string_view prefix) const noexcept {
        return find_node_index(prefix) != invalid_index;
    }

    [[nodiscard]] std::vector<std::string> prefix_search_impl(std::string_view prefix, SizeT limit = SizeMax) const {
        UInt32 start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }

public:
    SFlatTries() : node_pool() {
        node_pool.reserve(1024);
        create_node();
    }
};

// 允许用户提供 memory resource
class SPmrFlatTries : public TriesBase<SPmrFlatTries> {
    friend class TriesBase<SPmrFlatTries>;

    struct Node {
        std::pmr::map<char, UInt32> children;
        bool is_end = false;
        explicit Node(std::pmr::memory_resource *resource) noexcept : children(resource) {}
    };

    std::pmr::vector<Node> node_pool;
    std::pmr::memory_resource *memory_resource;
    static constexpr UInt32 invalid_index = std::numeric_limits<UInt32>::max();

    UInt32 create_node() {
        node_pool.emplace_back(memory_resource);
        return static_cast<UInt32>(node_pool.size() - 1);
    }

    void dfs(UInt32 node_index, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;
        const Node &current_node = node_pool[node_index];
        if (current_node.is_end) {
            result.push_back(current_word);
            if (result.size() >= limit) return;
        }

        for (const auto &[ch, child_node_index] : current_node.children) {
            current_word.push_back(ch);
            dfs(child_node_index, current_word, result, limit);
            current_word.pop_back();
            if (result.size() >= limit) return;
        }
    }

    [[nodiscard]] UInt32 find_node_index(std::string_view prefix) const noexcept {
        UInt32 current_node_index = 0; // root
        for (char ch : prefix) {
            const Node &current_node = node_pool[current_node_index];
            auto it = current_node.children.find(ch);
            if (it == current_node.children.end()) return invalid_index;
            current_node_index = it->second;
        }
        return current_node_index;
    }

    void push_impl(std::string_view word) {
        UInt32 current_node_index = 0; // root
        for (char ch : word) {
            // 这里不应该使用引用，即便是单线程，create node 也会导致悬空引用
            auto it = node_pool[current_node_index].children.find(ch);
            if (it == node_pool[current_node_index].children.end()) {
                UInt32 child_node_index = create_node();
                node_pool[current_node_index].children.try_emplace(ch, child_node_index);
                current_node_index = child_node_index;
            } else {
                current_node_index = it->second;
            }
        }
        node_pool[current_node_index].is_end = true;
    }

    [[nodiscard]] bool contains_impl(std::string_view word) const noexcept {
        UInt32 node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    bool erase_impl(std::string_view word) {
        UInt32 current_node_index = 0; // root
        std::pmr::vector<std::pair<UInt32, char>> path(memory_resource);
        path.reserve(word.size());
        for (char ch : word) {
            Node &current_node = node_pool[current_node_index];
            auto it = current_node.children.find(ch);
            if (it == current_node.children.end()) return false;
            path.emplace_back(current_node_index, ch);
            current_node_index = it->second;
        }

        Node &terminal_node = node_pool[current_node_index];
        if (!terminal_node.is_end) return false;
        terminal_node.is_end = false;
        if (!terminal_node.children.empty()) return true;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            UInt32 parent_index = it->first;
            char edge = it->second;
            Node &parent_node = node_pool[parent_index];
            auto child_it = parent_node.children.find(edge);
            if (child_it == parent_node.children.end()) [[unlikely]]
                break;

            UInt32 child_node_index = child_it->second;
            const Node &child_node = node_pool[child_node_index];
            if (child_node.is_end || !child_node.children.empty()) break;
            parent_node.children.erase(child_it);
        }
        return true;
    }

    [[nodiscard]] bool contains_starts_with_impl(std::string_view prefix) const noexcept {
        return find_node_index(prefix) != invalid_index;
    }

    [[nodiscard]] std::vector<std::string> prefix_search_impl(std::string_view prefix, SizeT limit = SizeMax) const {
        UInt32 start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }

public:
    explicit SPmrFlatTries(std::pmr::memory_resource *resource) : node_pool(resource), memory_resource(resource) {
        node_pool.reserve(1024);
        create_node();
    }
};

/*
 * @brief 允许用户提供 memory resource ，指定节点中存储孩子的数据结构，是否复用内存池中死亡的节点
 * @tparam ChildrenStorageType 节点中存储孩子的数据结构
 * @tparam ReuseDeadNodes 是否复用内存池中的死亡节点，启用可以节省内存，但降低缓存命中率
 */
template <TriesChildrenStorage ChildrenStorageType, bool ReuseDeadNodes = false>
class SPmrArrayTries : public TriesBase<SPmrArrayTries<ChildrenStorageType, ReuseDeadNodes>> {
    friend TriesBase<SPmrArrayTries<ChildrenStorageType, ReuseDeadNodes>>;

    using IndexType = UInt32;

    static constexpr IndexType invalid_index = std::numeric_limits<UInt32>::max();

    struct Node {
        ChildrenStorageType children;
        bool is_end = false;
    };

    std::pmr::vector<Node> node_pool;
    TriesFreeList<IndexType, ReuseDeadNodes> free_list;
    std::pmr::memory_resource *memory_resource;

    IndexType create_node() {
        if constexpr (ReuseDeadNodes) {
            if (!free_list.empty()) {
                IndexType reused_index = free_list.pop();
                Node &node = node_pool[reused_index];
                node.children.reset();
                node.is_end = false;
                return reused_index;
            }
        }
        node_pool.emplace_back();
        return static_cast<IndexType>(node_pool.size() - 1);
    }

    void dfs(IndexType node_index, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;
        const Node &node = node_pool[node_index];
        if (node.is_end) {
            result.push_back(current_word);
            if (result.size() >= limit) return;
        }
        if (node.children.empty()) return;

        node.children.for_each_child([this, &current_word, &result, limit](UChar key, IndexType child_node_index) {
            if (child_node_index == invalid_index) return; // defense
            if (result.size() >= limit) return;
            current_word.push_back(static_cast<char>(key));
            dfs(child_node_index, current_word, result, limit);
            current_word.pop_back();
        });
    }

    [[nodiscard]] IndexType find_node_index(std::string_view prefix) const noexcept {
        IndexType current_node_index = 0;
        for (char ch : prefix) {
            UChar key = CastUChar(ch);
            const Node &current_node = node_pool[current_node_index];
            IndexType child_node_index = current_node.children.get(key);
            if (child_node_index == invalid_index) return invalid_index;
            current_node_index = child_node_index;
        }
        return current_node_index;
    }

    void push_impl(std::string_view word) {
        IndexType current_node_index = 0;
        for (char ch : word) {
            UChar key = CastUChar(ch);
            // 不要使用引用，create node 会导致 reallocation
            IndexType child_node_index = node_pool[current_node_index].children.get(key);
            if (child_node_index == invalid_index) {
                child_node_index = create_node();
                node_pool[current_node_index].children.set(key, child_node_index);
            }
            current_node_index = child_node_index;
        }
        node_pool[current_node_index].is_end = true;
    }

    [[nodiscard]] bool contains_impl(std::string_view word) const noexcept {
        IndexType node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    bool erase_impl(std::string_view word) {
        IndexType current_node_index = 0;
        std::pmr::vector<std::pair<IndexType, UChar>> path(memory_resource);
        path.reserve(word.size());

        for (char ch : word) {
            UChar key = CastUChar(ch);
            Node &current_node = node_pool[current_node_index];
            IndexType child_node_index = current_node.children.get(key);
            if (child_node_index == invalid_index) return false;
            path.emplace_back(current_node_index, key);
            current_node_index = child_node_index;
        }

        Node &terminal_node = node_pool[current_node_index];
        if (!terminal_node.is_end) return false;
        terminal_node.is_end = false;
        if (!terminal_node.children.empty()) return true;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            IndexType parent_node_index = it->first;
            UChar child_key = it->second;
            Node &parent_node = node_pool[parent_node_index];
            IndexType child_node_index = parent_node.children.get(child_key);
            if (child_node_index == invalid_index) [[unlikely]]
                break;

            const Node &child_node = node_pool[child_node_index];
            if (child_node.is_end || !child_node.children.empty()) break;
            parent_node.children.set(child_key, invalid_index);
            if constexpr (ReuseDeadNodes) free_list.push(child_node_index);
        }
        return true;
    }

    [[nodiscard]] bool contains_starts_with_impl(std::string_view prefix) const noexcept {
        return find_node_index(prefix) != invalid_index;
    }

    [[nodiscard]] std::vector<std::string> prefix_search_impl(std::string_view prefix, SizeT limit = SizeMax) {
        IndexType start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }

public:
    explicit SPmrArrayTries(std::pmr::memory_resource *resource)
        : node_pool(resource), free_list(resource), memory_resource(resource) {
        node_pool.reserve(1024);
        create_node();
    }
};

template <TriesChildrenStorage ChildrenStorageType> using SPmrArrayTriesFL = SPmrArrayTries<ChildrenStorageType, true>;