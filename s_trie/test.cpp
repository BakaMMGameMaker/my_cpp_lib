// test.cpp
#include "salias.h"
#include "strie.hpp"
#include "stries_children_storage.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory_resource>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using STrieNoReuse = STrie<HybridDynamicChildren<UInt32, 256, 16>, false>;
using STrieWithReuse = STrie<HybridDynamicChildren<UInt32, 256, 16>, true>;
using STrieNoReuseKV = STrie<HybridDynamicChildren<UInt32, 256, 16>, false, int>;
using STrieWithReuseKV = STrie<HybridDynamicChildren<UInt32, 256, 16>, true, int>;

static const std::vector<std::string> kWords{
    "app", "apple", "apply", "banana", "band", "bandana", "cat", "car", "dog",
};

static bool contains(const std::vector<std::string> &v, std::string_view s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

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
    for (const auto &w : kWords) { trie.push(w); }

    assert(!trie.empty());
    assert(trie.size() == kWords.size());

    // contains 测试
    for (const auto &w : kWords) { assert(trie.contains(w)); }
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
        assert(trie.size() == kWords.size() - 1);
    }

    std::cout << "[OK] erase / word_count\n";
    std::cout << "active_node_count = " << trie.active_node_count() << "\n";
}

// 多线程测试（ValueType = void），测读写锁是否稳当
template <typename TrieType> void run_multithread_tests_void(TrieType &trie) {
    std::cout << "===== Multithread tests (void) =====\n";

    // 预热：插入一批单词，避免线程一启动就疯狂触发扩容，
    for (int i = 0; i < 1000; ++i) { trie.push("warmup_" + std::to_string(i)); }

    std::atomic<bool> stop{false};

    auto reader_task = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            (void)trie.contains("apple");
            (void)trie.contains_starts_with("ban");
            (void)trie.prefix_search("app", 8);
            (void)trie.longest_prefix_of("application");
        }
    };

    auto writer_task = [&]() {
        for (int i = 0; i < 2000; ++i) {
            trie.push("tword_" + std::to_string(i));
            if (i % 3 == 0) { trie.erase("tword_" + std::to_string(i / 2)); }
        }
    };

    std::thread r1(reader_task);
    std::thread r2(reader_task);
    std::thread r3(reader_task);
    std::thread w1(writer_task);
    std::thread w2(writer_task);

    w1.join();
    w2.join();
    stop.store(true, std::memory_order_relaxed);
    r1.join();
    r2.join();
    r3.join();

    // 简单 sanity check：随便查几个单词，确保调用没炸
    assert(trie.contains_starts_with("app"));
    (void)trie.prefix_search("tword_", 5);

    std::cout << "[OK] multithread (void)\n";
}

// 有值版本测试：insert_or_assign / find / prefix_search_with_value
template <typename KVTrie> void run_kv_tests(KVTrie &trie) {
    std::cout << "===== KV basic tests =====\n";

    assert(trie.empty());
    assert(trie.size() == 0);

    bool is_new = trie.insert_or_assign("app", 1);
    assert(is_new);
    assert(trie.size() == 1);

    // 覆盖已有 key
    is_new = trie.insert_or_assign("app", 2);
    assert(!is_new);
    assert(trie.size() == 1);

    // 再插入几个
    trie.insert_or_assign("apple", 3);
    trie.insert_or_assign("apply", 4);
    trie.insert_or_assign("banana", 5);
    trie.insert_or_assign("band", 6);
    trie.insert_or_assign("bandana", 7);

    assert(!trie.empty());
    assert(trie.size() == 6); // "app" + "apple" + "apply" + "banana" + "band" + "bandana"

    // find / const find
    {
        int *p = trie.find("app");
        assert(p && *p == 2);
        *p = 42; // 通过非 const 指针改值

        const auto &ctrie = trie;
        const int *cp = ctrie.find("app");
        assert(cp && *cp == 42);

        assert(trie.find("zzz") == nullptr);
        assert(ctrie.find("zzz") == nullptr);
    }

    // 迭代器遍历 / const_迭代器 / 引用语义测试
    {
        // 非 const iterator：遍历所有键
        std::vector<std::string> keys;
        for (auto it = trie.begin(); it != trie.end(); ++it) {
            auto kv = *it; // kv.second 是对内部值的引用（T&）
            keys.push_back(kv.first);
        }

        std::sort(keys.begin(), keys.end());
        std::vector<std::string> expected{
            "app", "apple", "apply", "banana", "band", "bandana",
        };
        std::sort(expected.begin(), expected.end());
        assert(keys == expected);

        {
            auto it = trie.begin();
            auto end_it = trie.end();
            bool found = false;
            for (; it != end_it; ++it) {
                auto kv = *it;
                if (kv.first == "app") {
                    found = true;
                    break;
                }
            }
            assert(found);
        }

        // const_iterator：只读遍历
        {
            const auto &ctrie = trie;
            std::vector<std::string> ckeys;
            for (auto it = ctrie.begin(); it != ctrie.end(); ++it) {
                auto kv = *it; // pair<string, const int&>
                ckeys.push_back(kv.first);
            }
            std::sort(ckeys.begin(), ckeys.end());
            assert(ckeys == expected);
        }
    }

    // prefix_search_with_value
    {
        auto res = trie.prefix_search_with_value("app");
        std::cout << "prefix_search_with_value(\"app\") [size = " << res.size() << "]: ";
        for (auto &kv : res) { std::cout << "(\"" << kv.first << "\", " << kv.second << ") "; }
        std::cout << "\n";

        assert(res.size() >= 3);
    }

    // prefix_search_with_value
    {
        auto res = trie.prefix_search_with_value("app");
        std::cout << "prefix_search_with_value(\"app\") [size = " << res.size() << "]: ";
        for (auto &kv : res) { std::cout << "(\"" << kv.first << "\", " << kv.second << ") "; }
        std::cout << "\n";

        assert(res.size() >= 3);
    }

    // 带 ScoreFunc 的 prefix_search_with_value：长度越短分越高
    {
        auto score = [](std::string_view s) noexcept { return static_cast<int>(100 - s.size()); };

        auto top2 = trie.prefix_search_with_value("ban", score, 2);
        std::cout << "prefix_search_with_value(\"ban\", score, limit=2) [size = " << top2.size() << "]: ";
        for (auto &kv : top2) { std::cout << "(\"" << kv.first << "\", " << kv.second << ") "; }
        std::cout << "\n";

        assert(!top2.empty());
        for (const auto &kv : top2) { assert(kv.first.rfind("ban", 0) == 0); }
    }

    // erase 也要在 ValueType != void 时正常工作
    {
        bool erased = trie.erase("band");
        assert(erased);
        assert(!trie.contains("band"));
        assert(trie.contains("bandana"));

        bool erased_again = trie.erase("band");
        assert(!erased_again);
    }

    std::cout << "[OK] KV basic tests\n";
}

// 有值版本的多线程测试：writer 用 insert_or_assign，reader 用 find / prefix_search_with_value
template <typename KVTrie> void run_multithread_tests_kv(KVTrie &trie) {
    std::cout << "===== Multithread tests (KV) =====\n";

    // 预热几条记录
    trie.insert_or_assign("base_app", 1);
    trie.insert_or_assign("base_apple", 2);
    trie.insert_or_assign("base_banana", 3);

    std::atomic<bool> stop{false};

    auto reader_task = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            (void)trie.find("base_app");
            (void)trie.prefix_search_with_value("base", 8);
        }
    };

    auto writer_task = [&]() {
        for (int i = 0; i < 2000; ++i) {
            trie.insert_or_assign("kv_" + std::to_string(i), i);
            if (i % 5 == 0) { trie.erase("kv_" + std::to_string(i / 2)); }
        }
    };

    std::thread r1(reader_task);
    std::thread r2(reader_task);
    std::thread w1(writer_task);
    std::thread w2(writer_task);

    w1.join();
    w2.join();
    stop.store(true, std::memory_order_relaxed);
    r1.join();
    r2.join();

    // 简单 sanity check
    const auto &ctrie = trie;
    const int *cp = ctrie.find("base_app");
    assert(cp != nullptr);

    std::cout << "[OK] multithread (KV)\n";
}

// 序列化 / 反序列化测试（仅 ValueType = void）
void run_serialize_tests(std::pmr::memory_resource *mr) {
    std::cout << "===== serialize / deserialize tests (void) =====\n";

    {
        STrieNoReuse trie(mr, 128);
        for (const auto &w : kWords) { trie.push(w); }

        std::ofstream ofs("trie.bin", std::ios::binary);
        if (!ofs) throw std::runtime_error("failed to open file for write");
        trie.serialize(ofs);
    }

    {
        std::ifstream ifs("trie.bin", std::ios::binary);
        if (!ifs) throw std::runtime_error("failed to open file for read");
        STrieNoReuse trie(mr, 128);
        trie.deserialize(ifs);

        // 验证几个典型单词
        assert(trie.contains("app"));
        assert(trie.contains("apple"));
        assert(trie.contains("banana"));
        assert(!trie.contains("zzz"));
    }

    std::cout << "[OK] serialize / deserialize\n";
}

int main() {
    std::pmr::monotonic_buffer_resource mr;

    {
        std::cout << "===== STrie<HybridDynamicChildren, false, void> =====\n";
        STrieNoReuse trie(&mr, 128);
        run_basic_tests(trie);
        run_multithread_tests_void(trie);
    }

    {
        std::cout << "\n===== STrie<HybridDynamicChildren, true, void> =====\n";
        STrieWithReuse trie(&mr, 128);
        run_basic_tests(trie);
        run_multithread_tests_void(trie);
    }

    // 序列化 / 反序列化
    run_serialize_tests(&mr);

    {
        std::cout << "\n===== STrie<HybridDynamicChildren, false, int> (KV) =====\n";
        STrieNoReuseKV trie(&mr, 128);
        run_kv_tests(trie);
        run_multithread_tests_kv(trie);
    }

    {
        std::cout << "\n===== STrie<HybridDynamicChildren, true, int> (KV) =====\n";
        STrieWithReuseKV trie(&mr, 128);
        run_kv_tests(trie);
        run_multithread_tests_kv(trie);
    }

    {
        STrieNoReuseKV trie;
        trie.insert_or_assign("apple", 114);

        trie.for_each_with_prefix("a", [](const std::string &word, int &value) {
            if (!word.empty() && word.front() == 'a' && word.back() == 'e') {
                value = 999; // 随便改
            }
        });

        assert(*trie.find("apple") == 999);
    }

    std::cout << "\nAll tests passed.\n";
    return 0;
}
