#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>

class STries {
    struct Node {
        // 使用 map 来确保 children 有序，dfs 结果明确
        std::map<char, std::unique_ptr<Node>> children;
        bool is_end = false;
    };
    std::unique_ptr<Node> root = std::make_unique<Node>();

    // 深度优先搜索收集结果
    void dfs(const Node *node, std::string &current_word, std::vector<std::string> &result,
             std::size_t limit) const noexcept {
        if (result.size() >= limit) return;
        if (node->is_end) result.push_back(current_word);
        if (result.size() >= limit) return;
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
    STries(STries &&) = default;
    STries &operator=(STries &&) = default;

    // 添加新的单词
    void push(std::string_view word) {
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
        auto node = find_node(word);
        return node && node->is_end;
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        auto current_node = root.get();
        std::vector<std::pair<Node *, char>> path; // 记录路径
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
            auto child = parent->children[edge].get();
            if (child->is_end || !child->children.empty()) break;
            // 不作为某单词的结尾，也没有任何孩子（意味着不作为任何单词的前缀部分）
            parent->children.erase(edge);
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept { return find_node(prefix); }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, std::size_t limit = SIZE_MAX) const {
        auto start_node = find_node(prefix);
        if (!start_node) return {};

        std::vector<std::string> result;
        std::string current_word(prefix);
        dfs(start_node, current_word, result, limit);
        return result;
    }
};

class SFlatTries {
    struct Node {
        std::map<char, std::uint32_t> children;
        bool is_end = false;
    };

    std::vector<Node> node_pool; // index 是在池中的下标
    static constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t create_node() {
        node_pool.emplace_back();
        return static_cast<std::uint32_t>(node_pool.size() - 1);
    }

    void dfs(std::uint32_t node_index, std::string &current_word, std::vector<std::string> &result,
             std::size_t limit) const noexcept {
        if (result.size() >= limit) return;
        // 不要持有引用，特别是在多线程查找情况下
        // const auto &node = node_pool[node_index];
        if (node_pool[node_index].is_end) result.push_back(current_word);
        if (result.size() >= limit) return;
        for (const auto &[ch, child_index] : node_pool[node_index].children) {
            current_word.push_back(ch);
            dfs(child_index, current_word, result, limit);
            current_word.pop_back();
            if (result.size() >= limit) return;
        }
    }

    // 定位前缀对应节点索引
    [[nodiscard]] std::uint32_t find_node_index(std::string_view prefix) const noexcept {
        std::uint32_t current_node_index = 0; // root
        for (char ch : prefix) {
            auto it = node_pool[current_node_index].children.find(ch);
            if (it == node_pool[current_node_index].children.end()) return invalid_index;
            current_node_index = it->second;
        }
        return current_node_index;
    }

public:
    SFlatTries() {
        node_pool.reserve(1024);
        create_node();
    }
    SFlatTries(const SFlatTries &) = default;
    SFlatTries &operator=(const SFlatTries &) = default;
    SFlatTries(SFlatTries &&) = default;
    SFlatTries &operator=(SFlatTries &&) = default;

    // 添加新的单词
    void push(std::string_view word) {
        std::uint32_t current_node_index = 0; // root
        for (char ch : word) {
            auto it = node_pool[current_node_index].children.find(ch);
            if (it == node_pool[current_node_index].children.end()) {
                auto new_child_index = create_node(); // 此处可能导致 reallocate
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
        auto node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        std::uint32_t current_node_index = 0; // root
        std::vector<std::pair<std::uint32_t, char>> path;
        for (char ch : word) {
            auto &current_node = node_pool[current_node_index];
            auto it = current_node.children.find(ch);
            if (it == current_node.children.end()) return false;
            path.emplace_back(current_node_index, ch);
            current_node_index = it->second;
        }

        auto &terminal_node = node_pool[current_node_index];
        if (!terminal_node.is_end) return false;
        terminal_node.is_end = false;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            auto parent_index = it->first;
            char edge = it->second;
            auto &parent_node = node_pool[parent_index];
            auto child_it = parent_node.children.find(edge);
            auto child_index = child_it->second;
            const auto &child_node = node_pool[child_index];
            if (child_node.is_end || !child_node.children.empty()) break;
            parent_node.children.erase(child_it);
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        return find_node_index(prefix) != invalid_index;
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, std::size_t limit = SIZE_MAX) const {
        auto start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};
        std::vector<std::string> result;
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }
};

class SPmrFlatTries {
    struct Node {
        std::pmr::map<char, std::uint32_t> children;
        bool is_end = false;
        explicit Node(std::pmr::memory_resource *resource) : children(resource) {}
    };

    std::pmr::vector<Node> node_pool;
    std::pmr::memory_resource *memory_resource;
    static constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t create_node() {
        node_pool.emplace_back(memory_resource);
        return static_cast<std::uint32_t>(node_pool.size() - 1);
    }

    void dfs(std::uint32_t node_index, std::string &current_word, std::vector<std::string> &result,
             std::size_t limit) const noexcept {
        if (result.size() >= limit) return;
        if (node_pool[node_index].is_end) result.push_back(current_word);
        if (result.size() >= limit) return;

        for (const auto &[ch, child_index] : node_pool[node_index].children) {
            current_word.push_back(ch);
            dfs(child_index, current_word, result, limit);
            current_word.pop_back();
            if (result.size() >= limit) return;
        }
    }

    [[nodiscard]] std::uint32_t find_node_index(std::string_view prefix) const noexcept {
        std::uint32_t current_node_index = 0; // root
        for (char ch : prefix) {
            const auto &current_node = node_pool[current_node_index];
            auto it = current_node.children.find(ch);
            if (it == current_node.children.end()) return invalid_index;
            current_node_index = it->second;
        }
        return current_node_index;
    }

public:
    explicit SPmrFlatTries(std::pmr::memory_resource *resource = std::pmr::get_default_resource())
        : node_pool(resource), memory_resource(resource) {
        node_pool.reserve(1024);
        create_node();
    }

    SPmrFlatTries(const SPmrFlatTries &) = delete;
    SPmrFlatTries &operator=(const SPmrFlatTries &) = delete;
    SPmrFlatTries(SPmrFlatTries &&) = default;
    SPmrFlatTries &operator=(SPmrFlatTries &&) = default;

    // 添加新的单词
    void push(std::string_view word) {
        std::uint32_t current_node_index = 0; // root
        for (char ch : word) {
            auto it = node_pool[current_node_index].children.find(ch);
            if (it == node_pool[current_node_index].children.end()) {
                auto new_child_index = create_node();
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
        auto node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end;
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        std::uint32_t current_node_index = 0; // root
        std::vector<std::pair<std::uint32_t, char>> path;
        for (char ch : word) {
            auto &current_node = node_pool[current_node_index];
            auto it = current_node.children.find(ch);
            if (it == current_node.children.end()) return false;
            path.emplace_back(current_node_index, ch);
            current_node_index = it->second;
        }

        auto &terminal_node = node_pool[current_node_index];
        if (!terminal_node.is_end) return false;
        terminal_node.is_end = false;

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            auto parent_index = it->first;
            char edge = it->second;
            auto &parent_node = node_pool[parent_index];
            auto child_it = parent_node.children.find(edge);
            auto child_index = child_it->second;
            const auto &child_node = node_pool[child_index];
            if (child_node.is_end || !child_node.children.empty()) break;
            parent_node.children.erase(child_it);
        }
        return true;
    }

    // 查找树中是否有以 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        return find_node_index(prefix) != invalid_index;
    }

    // 获取所有以 prefix 为前缀的单词，最多 limit 个
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, std::size_t limit = SIZE_MAX) const {
        auto start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};
        std::vector<std::string> result;
        std::string current_word(prefix);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }
};