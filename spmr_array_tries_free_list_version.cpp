#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

using SizeT = std::size_t;
using UInt32 = std::uint32_t;
using UChar = unsigned char;

static constexpr SizeT SizeMax = std::numeric_limits<SizeT>::max();

class SPmrArrayTries {
    static constexpr SizeT children_capacity = 256;
    static constexpr UInt32 invalid_index = std::numeric_limits<UInt32>::max();

    struct Node {
        std::array<UInt32, children_capacity> children;
        SizeT active_count;
        bool is_end;
        Node() noexcept : children(), active_count(0), is_end(false) {
            children.fill(invalid_index);
        }
    };

    std::pmr::vector<Node> node_pool;
    // free list for reusable node indices
    std::pmr::vector<UInt32> free_node_indices;
    mutable std::shared_mutex shared_mutex;

    static constexpr SizeT uchar_to_index(UChar uch) noexcept {
        return static_cast<SizeT>(uch);
    }

    UInt32 create_node() {
        if (!free_node_indices.empty()) {
            UInt32 reused_index = free_node_indices.back();
            free_node_indices.pop_back();
            node_pool[reused_index] = Node{}; // reset node state
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
            UChar uch = static_cast<UChar>(ch);
            const Node &current_node = node_pool[current_node_index];
            SizeT child_slot = uchar_to_index(uch);
            UInt32 child_index = current_node.children[child_slot];
            if (child_index == invalid_index) return invalid_index;
            current_node_index = child_index;
        }
        return current_node_index;
    }

public:
    explicit SPmrArrayTries(std::pmr::memory_resource *resource)
        : node_pool(resource), free_node_indices(resource), shared_mutex() {
        node_pool.reserve(1024);
        create_node(); // root
    }

    SPmrArrayTries(const SPmrArrayTries &) = delete;
    SPmrArrayTries &operator=(const SPmrArrayTries &) = delete;
    SPmrArrayTries(SPmrArrayTries &&) = delete;
    SPmrArrayTries &operator=(SPmrArrayTries &&) = delete;

    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        UInt32 current_node_index = 0;
        for (char ch : word) {
            UChar uch = static_cast<UChar>(ch);
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
            UChar uch = static_cast<UChar>(ch);
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

        for (auto reverse_iterator = path.rbegin(); reverse_iterator != path.rend(); ++reverse_iterator) {
            UInt32 parent_index = reverse_iterator->first;
            SizeT child_slot = reverse_iterator->second;
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

    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return find_node_index(prefix) != invalid_index;
    }

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
