#pragma once
#include "salias.h"
#include "sutils.hpp"
#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

// 原生的链式结构 Tries
class STries {
    struct Node {
        // 使用 map 来确保 children 有序，dfs 结果明确
        std::map<char, std::unique_ptr<Node>> children;
        bool is_end = false;
    };
    std::unique_ptr<Node> root = std::make_unique<Node>();
    mutable std::shared_mutex shared_mutex; // 读写锁：支持多线程读，写独占

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
        auto current_node = root.get();
        for (char ch : prefix) {
            auto it = current_node->children.find(ch);
            if (it == current_node->children.end()) return nullptr;
            current_node = it->second.get();
        }
        return current_node;
    }

public:
    STries() = default;
    STries(const STries &) = delete;
    STries &operator=(const STries &) = delete;
    STries(STries &&) = delete;
    STries &operator=(STries &&) = delete;

    // 添加新的单词
    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex); // 写独占
        auto current_node = root.get();
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

    // 查找是否包含给定单词
    [[nodiscard]] bool contains(std::string_view word) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex); // 多线程读
        auto node = find_node(word);
        return node && node->is_end;
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        auto current_node = root.get();
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
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            auto parent = it->first;
            char edge = it->second;
            auto child_it = parent->children.find(edge);
            if (child_it == parent->children.end()) break;
            auto child = child_it->second.get();
            if (child->is_end || !child->children.empty()) break;
            // 不作为某单词的结尾，也没有任何孩子（意味着不作为任何单词的前缀部分）
            parent->children.erase(child_it);
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return find_node(prefix);
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, SizeT limit = SizeMax) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        auto start_node = find_node(prefix);
        if (!start_node) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node, current_word, result, limit);
        return result;
    }
};

// 使用对象池存储节点
class SFlatTries {
    struct Node {
        std::map<char, UInt32> children;
        bool is_end = false;
    };

    std::vector<Node> node_pool; // index 是在池中的下标
    static constexpr UInt32 invalid_index = std::numeric_limits<UInt32>::max();
    mutable std::shared_mutex shared_mutex;

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

        for (const auto &[ch, child_index] : current_node.children) {
            current_word.push_back(ch);
            dfs(child_index, current_word, result, limit);
            current_word.pop_back();
            if (result.size() >= limit) return;
        }
    }

    // 定位前缀对应节点索引
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

public:
    SFlatTries() : node_pool(), shared_mutex() {
        node_pool.reserve(1024);
        create_node();
    }
    SFlatTries(const SFlatTries &) = delete;
    SFlatTries &operator=(const SFlatTries &) = delete;
    SFlatTries(SFlatTries &&) = delete;
    SFlatTries &operator=(SFlatTries &&) = delete;

    // 添加新的单词
    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        UInt32 current_node_index = 0; // root
        for (char ch : word) {
            // 这里不应该使用引用，即便是单线程，create node 也会导致悬空引用
            auto it = node_pool[current_node_index].children.find(ch);
            if (it == node_pool[current_node_index].children.end()) {
                UInt32 new_child_index = create_node(); // 此处可能导致 reallocate
                node_pool[current_node_index].children.try_emplace(ch, new_child_index);
                current_node_index = new_child_index;
            } else {
                current_node_index = it->second;
            }
        }
        node_pool[current_node_index].is_end = true;
    }

    // 查找是否包含给定单词
    [[nodiscard]] bool contains(std::string_view word) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
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

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            UInt32 parent_index = it->first;
            char edge = it->second;
            Node &parent_node = node_pool[parent_index];
            auto child_it = parent_node.children.find(edge);
            if (child_it == parent_node.children.end()) break;
            UInt32 child_index = child_it->second;
            const Node &child_node = node_pool[child_index];
            if (child_node.is_end || !child_node.children.empty()) break;
            parent_node.children.erase(child_it);
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return find_node_index(prefix) != invalid_index;
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, SizeT limit = SizeMax) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }
};

// 允许用户提供 memory resource
class SPmrFlatTries {
    struct Node {
        std::pmr::map<char, UInt32> children;
        bool is_end = false;
        explicit Node(std::pmr::memory_resource *resource) noexcept : children(resource) {}
    };

    std::pmr::vector<Node> node_pool;
    std::pmr::memory_resource *memory_resource;
    static constexpr UInt32 invalid_index = std::numeric_limits<UInt32>::max();
    mutable std::shared_mutex shared_mutex;

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

        for (const auto &[ch, child_index] : current_node.children) {
            current_word.push_back(ch);
            dfs(child_index, current_word, result, limit);
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

public:
    explicit SPmrFlatTries(std::pmr::memory_resource *resource)
        : node_pool(resource), memory_resource(resource), shared_mutex() {
        node_pool.reserve(1024);
        create_node();
    }

    SPmrFlatTries(const SPmrFlatTries &) = delete;
    SPmrFlatTries &operator=(const SPmrFlatTries &) = delete;
    SPmrFlatTries(SPmrFlatTries &&) = delete;
    SPmrFlatTries &operator=(SPmrFlatTries &&) = delete;

    // 添加新的单词
    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        UInt32 current_node_index = 0; // root
        for (char ch : word) {
            // 这里不应该使用引用，即便是单线程，create node 也会导致悬空引用
            auto it = node_pool[current_node_index].children.find(ch);
            if (it == node_pool[current_node_index].children.end()) {
                UInt32 new_child_index = create_node();
                node_pool[current_node_index].children.try_emplace(ch, new_child_index);
                current_node_index = new_child_index;
            } else {
                current_node_index = it->second;
            }
        }
        node_pool[current_node_index].is_end = true;
    }

    // 查找是否包含给定的单词
    [[nodiscard]] bool contains(std::string_view word) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
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

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            UInt32 parent_index = it->first;
            char edge = it->second;
            Node &parent_node = node_pool[parent_index];
            auto child_it = parent_node.children.find(edge);
            if (child_it == parent_node.children.end()) break;
            UInt32 child_index = child_it->second;
            const Node &child_node = node_pool[child_index];
            if (child_node.is_end || !child_node.children.empty()) break;
            parent_node.children.erase(child_it);
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return find_node_index(prefix) != invalid_index;
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, SizeT limit = SizeMax) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }
};

// 仅支持 ASCII 但更快的 Tries，用 FixedArray 作为每个节点存储孩子的数据结构
class SPmrArrayTries {
    static constexpr SizeT children_capacity = 256;
    static constexpr UInt32 invalid_index = std::numeric_limits<UInt32>::max();

    struct Node {
        std::array<UInt32, children_capacity> children;
        SizeT active_count; // 活跃的孩子数，用于 O1 判断 has children
        bool is_end;
        Node() noexcept : children(), active_count(0), is_end(false) { children.fill(invalid_index); }
    };

    std::pmr::vector<Node> node_pool;
    // std::pmr::memory_resource *memory_resource; 暂时没用上
    mutable std::shared_mutex shared_mutex;

    static constexpr SizeT uchar_to_index(UChar uch) noexcept { return static_cast<SizeT>(uch); }

    UInt32 create_node() {
        node_pool.emplace_back();
        return static_cast<UInt32>(node_pool.size() - 1);
    }

    void dfs(UInt32 node_index, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;
        const Node &node = node_pool[node_index];
        if (node.is_end) {
            result.push_back(current_word);
            if (result.size() >= limit) return;
        }
        if (node.active_count == 0) return; // 避免空转 256

        for (SizeT child_slot = 0; child_slot < children_capacity; ++child_slot) {
            UInt32 child_index = node.children[child_slot];
            if (child_index == invalid_index) continue;

            current_word.push_back(static_cast<char>(child_slot)); // slot 本身就可理解为 ASCII 码
            dfs(child_index, current_word, result, limit);
            current_word.pop_back();

            if (result.size() >= limit) return;
        }
    }

    [[nodiscard]] UInt32 find_node_index(std::string_view prefix) const noexcept {
        UInt32 current_node_index = 0;
        for (char ch : prefix) {
            UChar uch = CastUChar(ch);
            const Node &current_node = node_pool[current_node_index];
            SizeT child_slot = uchar_to_index(uch);
            UInt32 child_index = current_node.children[child_slot];
            if (child_index == invalid_index) return invalid_index;
            current_node_index = child_index;
        }
        return current_node_index;
    }

public:
    explicit SPmrArrayTries(std::pmr::memory_resource *resource) : node_pool(resource), shared_mutex() {
        node_pool.reserve(1024);
        create_node();
    }

    SPmrArrayTries(const SPmrArrayTries &) = delete;
    SPmrArrayTries &operator=(const SPmrArrayTries &) = delete;
    SPmrArrayTries(SPmrArrayTries &&) = delete;
    SPmrArrayTries &operator=(SPmrArrayTries &&) = delete;

    // 添加新的单词
    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        UInt32 current_node_index = 0;
        for (char ch : word) {
            UChar uch = CastUChar(ch);
            // 不要使用引用，create node 会导致 reallocation
            SizeT child_slot = uchar_to_index(uch);
            UInt32 child_index = node_pool[current_node_index].children[child_slot];
            if (child_index == invalid_index) {
                child_index = create_node();
                node_pool[current_node_index].children[child_slot] = child_index;
                node_pool[current_node_index].active_count++;
            }
            current_node_index = child_index;
        }
        node_pool[current_node_index].is_end = true;
    }

    // 查找是否包含给定单词
    [[nodiscard]] bool contains(std::string_view word) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    bool erase(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        UInt32 current_node_index = 0;
        std::vector<std::pair<UInt32, SizeT>> path;
        path.reserve(word.size());

        for (char ch : word) {
            UChar uch = CastUChar(ch);
            Node &current_node = node_pool[current_node_index];
            SizeT child_slot = uchar_to_index(uch);
            UInt32 child_index = current_node.children[child_slot];
            if (child_index == invalid_index) return false;
            path.emplace_back(current_node_index, child_slot);
            current_node_index = child_index;
        }

        Node &terminal_node = node_pool[current_node_index];
        if (!terminal_node.is_end) return false;
        terminal_node.is_end = false;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            UInt32 parent_index = it->first;
            SizeT child_slot = it->second;
            Node &parent_node = node_pool[parent_index];
            UInt32 child_index = parent_node.children[child_slot];
            if (child_index == invalid_index) break;
            const Node &child_node = node_pool[child_index];
            if (child_node.is_end || child_node.active_count > 0) break;
            parent_node.children[child_slot] = invalid_index;
            parent_node.active_count--;
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return find_node_index(prefix) != invalid_index;
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, SizeT limit = SizeMax) {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }
};

// 优化：可以重复利用死亡节点
class SPmrCompactTries {
    static constexpr SizeT children_capacity = 256;
    static constexpr UInt32 invalid_index = std::numeric_limits<UInt32>::max();

    struct Node {
        std::array<UInt32, children_capacity> children;
        SizeT active_count;
        bool is_end;
        Node() noexcept : children(), active_count(0), is_end(false) { children.fill(invalid_index); }
        void reset() noexcept {
            children.fill(invalid_index);
            active_count = 0;
            is_end = false;
        }
    };

    std::pmr::vector<Node> node_pool;
    std::pmr::vector<UInt32> free_node_indices; // free list for reusable node indices
    mutable std::shared_mutex shared_mutex;

    static constexpr SizeT uchar_to_index(UChar uch) noexcept { return static_cast<SizeT>(uch); }

    UInt32 create_node() {
        if (!free_node_indices.empty()) {
            UInt32 reused_index = free_node_indices.back();
            free_node_indices.pop_back();
            node_pool[reused_index].reset();
            return reused_index;
        }
        node_pool.emplace_back();
        return static_cast<UInt32>(node_pool.size() - 1);
    }

    void dfs(UInt32 node_index, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;
        const Node &node = node_pool[node_index];
        if (node.is_end) {
            result.push_back(current_word);
            if (result.size() >= limit) return;
        }
        if (node.active_count == 0) return;

        for (SizeT child_slot = 0; child_slot < children_capacity; ++child_slot) {
            UInt32 child_index = node.children[child_slot];
            if (child_index == invalid_index) continue;

            current_word.push_back(static_cast<char>(child_slot));
            dfs(child_index, current_word, result, limit);
            current_word.pop_back();

            if (result.size() >= limit) return;
        }
    }

    [[nodiscard]] UInt32 find_node_index(std::string_view prefix) const noexcept {
        UInt32 current_node_index = 0;
        for (char ch : prefix) {
            UChar uch = CastUChar(ch);
            const Node &current_node = node_pool[current_node_index];
            SizeT child_slot = uchar_to_index(uch);
            UInt32 child_index = current_node.children[child_slot];
            if (child_index == invalid_index) return invalid_index;
            current_node_index = child_index;
        }
        return current_node_index;
    }

public:
    explicit SPmrCompactTries(std::pmr::memory_resource *resource)
        : node_pool(resource), free_node_indices(resource), shared_mutex() {
        node_pool.reserve(1024);
        create_node();
    }

    SPmrCompactTries(const SPmrCompactTries &) = delete;
    SPmrCompactTries &operator=(const SPmrCompactTries &) = delete;
    SPmrCompactTries(SPmrCompactTries &&) = delete;
    SPmrCompactTries &operator=(SPmrCompactTries &&) = delete;

    // 添加新的单词
    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        UInt32 current_node_index = 0;
        for (char ch : word) {
            UChar uch = CastUChar(ch);
            SizeT child_slot = uchar_to_index(uch);
            UInt32 child_index = node_pool[current_node_index].children[child_slot];
            if (child_index == invalid_index) {
                child_index = create_node();
                node_pool[current_node_index].children[child_slot] = child_index;
                node_pool[current_node_index].active_count++;
            }
            current_node_index = child_index;
        }
        node_pool[current_node_index].is_end = true;
    }

    // 查找是否包含给定单词
    [[nodiscard]] bool contains(std::string_view word) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        UInt32 current_node_index = 0;
        std::vector<std::pair<UInt32, SizeT>> path;
        path.reserve(word.size());

        for (char ch : word) {
            UChar uch = CastUChar(ch);
            Node &current_node = node_pool[current_node_index];
            size_t child_slot = uchar_to_index(uch);
            UInt32 child_index = current_node.children[child_slot];
            if (child_index == invalid_index) return false;
            path.emplace_back(current_node_index, child_slot);
            current_node_index = child_index;
        }

        Node &terminal_node = node_pool[current_node_index];
        if (!terminal_node.is_end) return false;
        terminal_node.is_end = false;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            UInt32 parent_index = it->first;
            SizeT child_slot = it->second;
            Node &parent_node = node_pool[parent_index];
            UInt32 child_index = parent_node.children[child_slot];
            if (child_index == invalid_index) break;
            const Node &child_node = node_pool[child_index];
            if (child_node.is_end || child_node.active_count > 0) break;
            parent_node.children[child_slot] = invalid_index;
            parent_node.active_count--;
            free_node_indices.push_back(child_index);
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return find_node_index(prefix) != invalid_index;
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, SizeT limit = SizeMax) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        UInt32 start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : static_cast<SizeT>(limit));
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }
};