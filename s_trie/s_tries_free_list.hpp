#pragma once
#include "salias.h"

// FreeList 的结构，用于开关死节点复用特性
template <typename IndexType, bool Enable> struct TriesFreeList;

template <typename IndexType> struct TriesFreeList<IndexType, false> {
    explicit TriesFreeList(std::pmr::memory_resource *) noexcept {}

    [[nodiscard]] bool empty() const noexcept { return true; }
    [[nodiscard]] SizeT size() const noexcept { return 0; }
    void push(IndexType) noexcept {}
    [[nodiscard]] IndexType pop() noexcept { return {}; }
};

template <typename IndexType> struct TriesFreeList<IndexType, true> {
    std::pmr::vector<IndexType> indices;
    explicit TriesFreeList(std::pmr::memory_resource *resource) : indices(resource) {}

    [[nodiscard]] bool empty() const noexcept { return indices.empty(); }
    [[nodiscard]] SizeT size() const noexcept { return indices.size(); }
    void push(IndexType index) { indices.push_back(index); }
    [[nodiscard]] IndexType pop() {
        IndexType index = indices.back();
        indices.pop_back();
        return index;
    }
};