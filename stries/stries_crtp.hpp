#pragma once
#include "salias.h"
#include <shared_mutex>
#include <string>
#include <string_view>

template <typename Derived>
concept DerivedTries = requires(Derived d, const Derived cd, std::string_view sv, SizeT limit) {
    { d.push_impl(sv) } -> std::same_as<void>;                                      // 添加单词
    { d.erase_impl(sv) } -> std::same_as<bool>;                                     // 移除单词
    { cd.size_impl() } -> std::same_as<SizeT>;                                      // 单词总数
    { cd.empty_impl() } -> std::same_as<bool>;                                      // 树是否为空
    { cd.active_node_count_impl() } -> std::same_as<SizeT>;                         // 逻辑活跃的节点数
    { cd.contains_impl(sv) } -> std::same_as<bool>;                                 // 是否包含某单词
    { cd.contains_starts_with_impl(sv) } -> std::same_as<bool>;                     // 是否包含以给定 sv 为前缀的单词
    { cd.prefix_search_impl(sv, limit) } -> std::same_as<std::vector<std::string>>; // 所有以 sv 为前缀的单词
    { cd.longest_prefix_of_impl(sv) } -> std::same_as<std::string_view>;            // 最长前缀匹配结果
};

// CRTP base
template <DerivedTries Derived> class TriesBase {
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
    // 获取单词总数
    [[nodiscard]] SizeT size() const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex_);
        return derived().size_impl();
    }

    // 树是否为空
    [[nodiscard]] bool empty() const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex_);
        return derived().empty_impl();
    }

    // 活跃节点总数（启用节点回收时值有意义）
    [[nodiscard]] SizeT active_node_count() const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex_);
        return derived().active_node_count_impl();
    }

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

    // 获取给定文本的所有前缀中在树中存在且作为完整单词的最长结果
    [[nodiscard]] std::string_view longest_prefix_of(std::string_view text) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex_);
        return derived().longest_prefix_of_impl(text);
    }
};