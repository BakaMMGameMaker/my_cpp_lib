// test_tries.cpp
#include "salias.h"
#include "strie.hpp"
#include "stries_children_storage.hpp"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory_resource>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using STrieNoReuse = STrie<HybridDynamicChildren<UInt32, 256, 16>, false>;
using STrieWithReuse = STrie<HybridDynamicChildren<UInt32, 256, 16>, true>;

// 简单工具：检查某个 vector 是否包含指定元素
static bool contains(const std::vector<std::string> &v, std::string_view s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

// 打印一个字符串数组（方便调试）
static void print_vector(std::string_view title, const std::vector<std::string> &v) {
    std::cout << title << " [size = " << v.size() << "]: ";
    for (const auto &s : v) { std::cout << '"' << s << "\" "; }
    std::cout << "\n";
}

template <Tries TrieType> void run_basic_tests(TrieType &trie) {
    std::cout << "===== Basic tests =====\n";

    // 初始状态
    assert(trie.empty());
    assert(trie.size() == 0);
    assert(trie.active_node_count() >= 1); // 至少有根节点

    // 插入一些单词
    const std::vector<std::string> words{"app", "apple", "apply", "banana", "band", "bandana", "cat", "car", "dog"};

    for (const auto &w : words) { trie.push(w); }

    assert(!trie.empty());
    assert(trie.size() == words.size());

    // contains 测试
    for (const auto &w : words) { assert(trie.contains(w)); }
    assert(!trie.contains("ape"));
    assert(!trie.contains("ban")); // 前缀不是完整单词

    // contains_starts_with 测试
    assert(trie.contains_starts_with("app"));
    assert(trie.contains_starts_with("ban"));
    assert(trie.contains_starts_with("ba"));
    assert(trie.contains_starts_with("c"));
    assert(!trie.contains_starts_with("zzz"));

    std::cout << "[OK] contains / contains_starts_with\n";

    // prefix_search
    {
        auto app_words = trie.prefix_search("app");
        print_vector("prefix_search(\"app\")", app_words);
        // 至少包含这三个
        assert(contains(app_words, "app"));
        assert(contains(app_words, "apple"));
        assert(contains(app_words, "apply"));
    }

    // prefix_search + limit
    {
        auto app_two = trie.prefix_search("app", 2);
        print_vector("prefix_search(\"app\", limit=2)", app_two);
        assert(app_two.size() <= 2);
        for (const auto &w : app_two) {
            assert(w.rfind("app", 0) == 0); // 必须以 "app" 开头
        }
    }

    std::cout << "[OK] prefix_search without score\n";

    // 带 ScoreFunc 的 prefix_search：用“越短分数越高”来排序
    {
        auto score = [](std::string_view s) noexcept {
            // 分数越大优先级越高：这里简单用长度的反比
            return static_cast<int>(100 - s.size());
        };

        auto top2 = trie.prefix_search("ban", score, 2);
        print_vector("prefix_search(\"ban\", score, limit=2)", top2);

        // 有结果，且都是以 "ban" 开头
        assert(!top2.empty());
        for (const auto &w : top2) { assert(w.rfind("ban", 0) == 0); }
    }

    std::cout << "[OK] prefix_search with score\n";

    // longest_prefix_of 测试
    {
        auto p1 = trie.longest_prefix_of("applepie");
        auto p2 = trie.longest_prefix_of("application");
        auto p3 = trie.longest_prefix_of("bandage");
        auto p4 = trie.longest_prefix_of("zzz");

        std::cout << "longest_prefix_of(\"applepie\")      = \"" << p1 << "\"\n";
        std::cout << "longest_prefix_of(\"application\")   = \"" << p2 << "\"\n";
        std::cout << "longest_prefix_of(\"bandage\")       = \"" << p3 << "\"\n";
        std::cout << "longest_prefix_of(\"zzz\")           = \"" << p4 << "\"\n";

        // 这里的期望依赖于你插入的单词
        assert(p1 == "apple");
        assert(p2 == "app");
        assert(p3 == "band"); // "bandage" 前缀里最长的是 "band"
        assert(p4.empty());
    }

    std::cout << "[OK] longest_prefix_of\n";

    // erase 测试
    {
        bool erased = trie.erase("band");
        assert(erased);
        assert(!trie.contains("band"));
        assert(trie.contains("bandana")); // 同前缀其他词不受影响

        // 再删一次应该失败
        bool erased_again = trie.erase("band");
        assert(!erased_again);

        // size 少了一个
        assert(trie.size() == words.size() - 1);
    }

    std::cout << "[OK] erase / word_count\n";
    std::cout << "active_node_count = " << trie.active_node_count() << "\n";
}

int main() {
    std::pmr::monotonic_buffer_resource mr;

    {
        std::cout << "===== STries<HybridDynamicChildren, false> =====\n";
        STrieNoReuse trie(&mr, 128);
        run_basic_tests(trie);
    }

    {
        std::cout << "\n===== STries<HybridDynamicChildren, true> =====\n";
        STrieWithReuse trie(&mr, 128);
        run_basic_tests(trie);
    }

    {
        STrieNoReuse trie(&mr, 128);
        std::ofstream ofs("trie.bin", std::ios::binary);
        if (!ofs) throw std::runtime_error("failed to open file for write");
        trie.serialize(ofs);
    }

    {
        std::ifstream ifs("trie.bin", std::ios::binary);
        if (!ifs) throw std::runtime_error("failed to open file for read");
        STrieNoReuse trie(&mr, 128);
        trie.deserialize(ifs);
    }

    std::cout << "\nAll tests passed.\n";
    return 0;
}
