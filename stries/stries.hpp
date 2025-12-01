#pragma once
#include "salias.h"
#include "stries_children_storage.hpp"
#include "stries_crtp.hpp"
#include "stries_free_list.hpp"
#include "sutils.hpp"
#include <limits>
#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>

/*
 * @brief 允许用户提供 memory resource ，指定节点中存储孩子的数据结构，是否复用内存池中死亡的节点
 * @tparam ChildrenStorageType 节点中存储孩子的数据结构
 * @tparam ReuseDeadNodes 是否复用内存池中的死亡节点，启用可以节省内存，但降低缓存命中率
 */
template <TriesChildrenStorage ChildrenStorageType, bool ReuseDeadNodes = false>
class STries : public TriesBase<STries<ChildrenStorageType, ReuseDeadNodes>> {
    friend TriesBase<STries<ChildrenStorageType, ReuseDeadNodes>>;

    using IndexType = UInt32;

    static constexpr IndexType invalid_index = std::numeric_limits<UInt32>::max();

    struct Node {
        ChildrenStorageType children;
        // 把 is end 压缩到 UInt32 的最高位上，免除 padding？
        // 没有必要，ChildrenStorageType 已经够大了，而且这么做会导致复杂性和可读性都更加恶劣
        bool is_end = false;
    };

    // TODO：deque 版本，但考虑到 deque 版本缓存比较不友好，暂时不考虑，更倾向于一次性 reserve 足够空间
    std::pmr::vector<Node> node_pool;                   // 节点内存池
    TriesFreeList<IndexType, ReuseDeadNodes> free_list; // 空闲节点列表
    std::pmr::memory_resource *memory_resource;         // 用户提供的内存资源
    SizeT word_count;                                   // 单词数量统计

    [[nodiscard]] SizeT size_impl() const noexcept { return word_count; }
    [[nodiscard]] bool empty_impl() const noexcept { return word_count == 0; }
    [[nodiscard]] SizeT active_node_count_impl() const noexcept { return node_pool.size() - free_list.size(); }

    IndexType create_node() {
        if constexpr (ReuseDeadNodes) {                   // 如果打算复用死亡节点
            if (!free_list.empty()) {                     // 如果空闲节点表不为空
                IndexType reused_index = free_list.pop(); // 弹出末尾的空闲节点对应的索引
                Node &node = node_pool[reused_index];     // 获取索引对应的空闲节点
                node.children.reset();                    // 重置空闲节点孩子状态
                node.is_end = false;                      // 重置空闲节点 is end 状态
                return reused_index;                      // 返回空闲接待你对应的索引
            }
        }
        node_pool.emplace_back();                            // 在对象池创建一个新的节点
        return static_cast<IndexType>(node_pool.size() - 1); // 返回新节点的索引
    }

    void dfs(IndexType node_index, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;       // 如果 result 已经到达 limit 直接返回
        const Node &node = node_pool[node_index]; // 获取当前节点
        if (node.is_end) {                        // 如果到当前节点构成完整单词
            result.push_back(current_word);       // 往 result 中加入当前完整单词
            if (result.size() >= limit) return;   // 再次检查 result 是否到达 limit
        }
        if (node.children.empty()) return; // 如果当前节点没有孩子，直接返回

        node.children.for_each_child([this, &current_word, &result, limit](UChar key, IndexType child_node_index) {
            if (child_node_index == invalid_index) return;      // 防御
            if (result.size() >= limit) return;                 // 到达 limit 返回
            current_word.push_back(static_cast<char>(key));     // 往当前单词中加入当前节点对应字符
            dfs(child_node_index, current_word, result, limit); // 递归 dfs
            current_word.pop_back();                            // 在当前单词弹出当前节点对应字符
        });
    }

    [[nodiscard]] IndexType find_node_index(std::string_view prefix) const noexcept {
        if (node_pool.empty()) [[unlikely]] // 防御
            return invalid_index;
        IndexType current_node_index = 0; // 根节点索引
        for (char ch : prefix) {
            UChar key = CastUChar(ch);
            const Node &current_node = node_pool[current_node_index];    // 当前节点
            IndexType child_node_index = current_node.children.get(key); // 孩子节点索引
            if (child_node_index == invalid_index) return invalid_index; // 孩子节点不存在，返回 invalid
            current_node_index = child_node_index;                       // 更新当前节点索引
        }
        return current_node_index;
    }

    void push_impl(std::string_view word) {
        IndexType current_node_index = 0;
        for (char ch : word) {
            UChar key = CastUChar(ch);
            // 不要使用引用，create node 会导致 reallocation
            IndexType child_node_index = node_pool[current_node_index].children.get(key);
            if (child_node_index == invalid_index) {                               // 如果孩子节点不存在
                child_node_index = create_node();                                  // 创建一个孩子节点
                node_pool[current_node_index].children.set(key, child_node_index); // 更新父节点信息
            }
            current_node_index = child_node_index;
        }
        Node &last_node = node_pool[current_node_index];
        if (last_node.is_end) return; // 插入的单词已经存在
        last_node.is_end = true;
        word_count++;
    }

    bool erase_impl(std::string_view word) {
        IndexType current_node_index = 0;                                    // 根节点索引
        std::pmr::vector<std::pair<IndexType, UChar>> path(memory_resource); // 记录删除路径 <父节点索引，孩子键>
        path.reserve(word.size());

        for (char ch : word) {
            UChar key = CastUChar(ch);
            Node &current_node = node_pool[current_node_index];          // 当前节点
            IndexType child_node_index = current_node.children.get(key); // 孩子节点索引
            if (child_node_index == invalid_index) return false;         // 单词不存在，删除失败
            path.emplace_back(current_node_index, key);                  // 新增 <当前节点，孩子键>
            current_node_index = child_node_index;                       // 更新当前节点索引
        }

        Node &last_node = node_pool[current_node_index]; // 单词末尾节点
        if (!last_node.is_end) return false;             // 单词在树中不构成完整单词，删除失败
        last_node.is_end = false;                        // 末尾节点不再为单词末尾节点
        if (!last_node.children.empty()) {               // 末尾节点有孩子，无需进一步删除，删除成功
            word_count--;
            return true;
        }

        // 回溯：把所有非单词末尾且没有孩子的无效节点都删除
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            IndexType parent_node_index = it->first;                          // 父节点索引
            UChar child_key = it->second;                                     // 孩子键
            Node &parent_node = node_pool[parent_node_index];                 // 父节点
            IndexType child_node_index = parent_node.children.get(child_key); // 孩子节点索引
            if (child_node_index == invalid_index) [[unlikely]]               // 防御
                break;
            const Node &child_node = node_pool[child_node_index];           // 孩子节点
            if (child_node.is_end || !child_node.children.empty()) break;   // 为单词末尾或有孩子，无需进一步删除
            parent_node.children.set(child_key, invalid_index);             // 更新父节点，删除孩子节点
            if constexpr (ReuseDeadNodes) free_list.push(child_node_index); // 更新空闲节点列表
        }
        word_count--;
        return true;
    }

    [[nodiscard]] bool contains_impl(std::string_view word) const noexcept {
        IndexType node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end; // 能找到对应节点且对应节点 is end 为真
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
        current_word.reserve(prefix.size() + 32);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }

    [[nodiscard]] std::string_view longest_prefix_of_impl(std::string_view text) {
        IndexType current_node_index = 0; // 根节点索引
        SizeT last_match_pos = 0;         // 最后匹配成功的位置
        bool found = false;
        for (SizeT i = 0; i < text.size(); ++i) {
            UChar key = CastUChar(text[i]);
            const Node &current_node = node_pool[current_node_index];    // 当前节点
            IndexType child_node_index = current_node.children.get(key); // 孩子节点索引
            if (child_node_index == invalid_index) break;                // 孩子节点不存在
            const Node &child_node = node_pool[child_node_index];        // 孩子节点
            current_node_index = child_node_index;
            if (!child_node.is_end) continue;
            last_match_pos = i + 1;
            found = true;
        }
        if (!found) return {};
        return text.substr(0, last_match_pos);
    }

public:
    explicit STries(std::pmr::memory_resource *resource, SizeT initial_capacity = 1024)
        : node_pool(resource), free_list(resource), memory_resource(resource), word_count(0) {
        node_pool.reserve(initial_capacity);
        create_node();
    }
};