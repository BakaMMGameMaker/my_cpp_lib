#include <stdexcept>
#include <utility>
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "s_alias.h"
#include "s_bucket_map_u32.hpp"
#include <cassert>
#include <iostream>

using map_t = mcl::bucket_map_u32<UInt32>;

static void test_construction() {
    // 默认构造
    map_t m1;
    assert(m1.capacity() == 2); // ceil 1 / 0.75 = 2
    assert(m1.size() == 0);
    assert(m1.empty());
    assert(m1.begin() == m1.end());
    assert(m1.cbegin() == m1.cend());

    // 参数构造
    map_t m2(0);
    assert(m2.capacity() == 0);
    assert(m2.capacity() == 0);
    assert(m2.size() == 0);
    assert(m2.empty());
    assert(m2.begin() == m2.end());
    assert(m2.cbegin() == m2.cend());
    assert(m1 == m2);

    // 初始化列表构造
    map_t m3{{2, 3}, {3, 4}, {4, 5}, {2, 1}};
    assert(m3.contains(2));
    assert(m3.contains(3));
    assert(m3.contains(4));
    assert(m3.find(2) != m3.end());
    auto it = m3.find(2);
    assert(it->second == 3);
    assert(m3.size() == 3);
    assert(!m3.empty());
    assert(m3.capacity() == 8); // 4 / 0.75

    // 拷贝构造
    map_t m4(m3);
    assert(m4.contains(2));
    assert(m4.contains(3));
    assert(m4.contains(4));

    assert(m4.find(3) != m4.end());
    auto it2 = m4.find(4);
    assert(it2->first == 4);
    assert(it2->second == 5);
    assert(!m4.empty());
    assert(m4.capacity() == 8); // == m3

    // 移动构造
    map_t m5(std::move(m3));
    assert(m5.contains(2));
    assert(m5.contains(3));
    assert(m5.contains(4));
    assert(m5.find(3) != m5.end());
    auto it3 = m5.find(4);
    assert(it3->first == 4);
    assert(it3->second == 5);
    assert(!m5.empty());
    assert(m5.capacity() == 8);
    assert(m3.empty());
    assert(m3.size() == 0);
    assert(m3.capacity() == 0);

    // 拷贝赋值
    map_t m6(16);
    assert(m6.empty());
    assert(m6.capacity() == 32);
    m6 = m5;
    assert(m6.contains(2));
    assert(m6.contains(3));
    assert(m6.contains(4));
    assert(m6.find(3) != m6.end());
    auto it4 = m6.find(4);
    assert(it4->first == 4);
    assert(it4->second == 5);
    assert(!m6.empty());
    assert(m6.capacity() == 32);
    assert(!m5.empty());
    assert(m5.size() == 3);
    assert(m5.capacity() == 8);
    assert(m5 == m6);

    // 移动赋值
    map_t m7(0);
    m7 = std::move(m6);
    assert(m7.contains(2));
    assert(m7.contains(3));
    assert(m7.contains(4));
    assert(m7.find(3) != m7.end());
    auto it5 = m7.find(4);
    assert(it5->first == 4);
    assert(it5->second == 5);
    assert(!m7.empty());
    assert(m7.capacity() == 32);
    assert(m6.empty());
    assert(m6.size() == 0);
    assert(m6.capacity() == 0);
}

static void test_load_factor() {
    map_t m;
    assert(m.load_factor() < 0.0001f);
    m.emplace(1, 2);
    assert(m.load_factor() > 0.0001f);
    m.max_load_factor(0.8f);
    assert(m.capacity() == 2);
    assert(m.max_load_factor() > 0.799f);
    assert(m.growth_limit() == 1);
}

static void test_clear() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}};
    assert(m.capacity() == 8);
    assert(m.size() == 3);
    m.clear();
    assert(m.capacity() == 8);
    assert(m.size() == 0);
}

static void test_reserve() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}};
    m.reserve(2);
    assert(m.capacity() == 8);
    assert(m.size() == 3);
    m.reserve(8);
    assert(m.capacity() == 16);
    assert(m.size() == 3);
    m.reserve(16);
    assert(m.capacity() == 32);
    assert(m.size() == 3);
    assert(m.contains(6));
    assert(m.find(4) != m.end());
    auto it = m.find(2);
    assert(it->second == 3);
    assert(!m.empty());
    assert(m.find(1) == m.end());
}

static void test_shrink_to_fit() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}, {3, 8}, {8, 10}};
    m.reserve(64);
    m.shrink_to_fit();
    assert(m.capacity() == 8);
    assert(m.size() == 5);

    for (UInt32 k = 12; k < 24; ++k) {
        m.try_emplace(k, k << 2);
        assert(m.contains(k));
        assert(m.find(k) != m.end());
        assert(m.find(k)->second == k << 2);
    }
    assert(m.capacity() == 32);
    assert(m.size() == 17);
    m.reserve(64);
    m.shrink_to_fit();
    assert(m.capacity() == 32);
    assert(m.size() == 17);

    for (UInt32 k = 12; k < 24; ++k) {
        auto [it, inserted] = m.emplace(k, k << 3);
        assert(it != m.end());
        assert(!inserted);
        assert(it->second == k << 2);
    }
    for (int k : {2, 3, 4, 6, 8}) { assert(m.contains(static_cast<UInt32>(k))); }
}

static void test_swap() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}, {3, 8}};
    for (UInt32 k = 11; k < 30; ++k) {
        m.insert_or_assign(k, k << 3);
        assert(m.contains(2));
        assert(m.contains(k));
        assert(m.find(k)->second == k << 3);
    }
    assert(m.capacity() == 32);
    map_t m2{{1, 2}};
    std::swap(m, m2);
    assert(m.capacity() == 2);
    assert(m.size() == 1);
    assert(m2.capacity() == 32);
    assert(m2.size() == 23);
    assert(m2.contains(2));
    assert(m2.find_exist(3)->second == 8);
    assert(m2.find(4)->second == 5);
    assert(m2.contains(6));
    for (UInt32 k = 11; k < 30; ++k) { assert(m2.find_exist(k)->second == k << 3); }
}

static void test_rehash() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}, {3, 8}};
    m.rehash(45);
    assert(m.capacity() == 64);
    m.rehash(32);
    assert(m.capacity() == 64);
    for (UInt32 k = 20; k < 100; ++k) {
        m.insert<mcl::no_return>({k, k << 2});
        assert(m.find(k) != m.end());
    }
    assert(m.size() == 84);
    assert(m.capacity() == 128);
    m.rehash(64);
    assert(m.capacity() == 128);
    m.rehash(128);
    assert(m.capacity() == 128);
    m.rehash(256);
    assert(m.capacity() == 256);
}

static void test_iterators() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}, {3, 8}};
    assert(m.begin() != m.end());
    assert(m.begin()->first == 2);
    assert(m.begin()->second == 3);
    assert(m.cbegin() != m.cend());
    SizeT cnt = 0;
    for (auto it = m.begin(); it != m.end(); ++it) {
        ++cnt;
        (void)it;
    }
    assert(cnt == 4);
    cnt = 0;
    for (auto kv : m) {
        ++cnt;
        (void)kv;
    }
    assert(cnt == 4);
    auto it = m.begin();
    ++it;
    m.clear();
    assert(m.begin() == m.end());
    assert(m.cbegin() == m.cend());
}

static void test_exist() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}, {3, 8}};
    assert(m.find_exist(2)->second == 3);
    assert(m.overwrite(2, 4u));
    assert(m.find_exist(2)->second == 4);
    assert(m.find_exist(10) == m.end());
    assert(m.at(4) == 5);
    assert(!m.overwrite(10, 20u));
    try {
        m.at(10);
    } catch (const std::out_of_range &e) { std::cout << e.what() << std::endl; }
}

static void test_erase() {
    map_t m{{2, 3}, {4, 5}, {6, 7}, {2, 5}, {3, 8}};
    assert(m.erase(2) == 1);
    assert(m.contains(3));
    assert(m.contains(4));
    assert(m.contains(6));
    assert(m.size() == 3);
    m.erase_exist(4);
    assert(!m.contains(4));
    auto it = m.find(6);
    m.erase(it);
    assert(!m.contains(6));
    try {
        m.erase_exist(10);
    } catch (const std::out_of_range &e) { std::cout << e.what() << std::endl; }
}

int main() {
    std::cout << "[bucket_map_u32] basic tests running...\n";

    test_construction();
    std::cout << "  construction test OK\n";

    test_load_factor();
    std::cout << "  load factor test OK\n";

    test_clear();
    std::cout << "  clear test OK\n";

    test_reserve();
    std::cout << "  reserve test OK\n";

    test_shrink_to_fit();
    std::cout << "  shrink to fit test OK\n";

    test_swap();
    std::cout << "  swap test OK\n";

    test_rehash();
    std::cout << "  rehash test OK\n";

    test_iterators();
    std::cout << "  iterators test OK\n";

    test_exist();
    std::cout << "  exist test OK\n";

    test_erase();
    std::cout << "  erase test OK\n";
}
