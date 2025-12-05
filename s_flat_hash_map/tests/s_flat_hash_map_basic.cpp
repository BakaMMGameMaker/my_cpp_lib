// s_flat_hash_map/tests/s_flat_hash_map_basic.cpp
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "s_flat_hash_map.hpp"
#include "salias.h"
#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

struct bad_hash_u32 {
    using is_transparent = void;
    SizeT operator()(UInt32) const noexcept { return 0xdeadbeefu; }
};

struct bad_hash_u64 {
    using is_transparent = void;
    SizeT operator()(UInt64) const noexcept { return 0xdeadbeefu; }
};

struct AllocStats {
    static inline Int64 alloc_count = 0;
    static inline Int64 dealloc_count = 0;

    static void reset() noexcept {
        alloc_count = 0;
        dealloc_count = 0;
    }
};

template <typename T> struct CountingAllocator {
    using value_type = T;
    CountingAllocator() noexcept = default;

    template <typename U> CountingAllocator(const CountingAllocator<U> &) noexcept {}

    T *allocate(SizeT n) {
        AllocStats::alloc_count += static_cast<Int64>(n);
        return std::allocator<T>{}.allocate(n);
    }

    void deallocate(T *p, SizeT n) noexcept {
        AllocStats::dealloc_count += static_cast<Int64>(n);
        std::allocator<T>{}.deallocate(p, n);
    }

    template <typename U> bool operator==(const CountingAllocator<U> &) const noexcept { return true; }
    template <typename U> bool operator!=(const CountingAllocator<U> &) const noexcept { return false; }
};

// 基础行为测试
static void test_basic_operations() {
    using map_t = mcl::flat_hash_map<UInt32, UInt32>;
    map_t m;
    assert(m.empty()); // 初始建表为空
    assert(m.size() == 0);
    assert(m.capacity() == 0); // 延迟分配

    // insert / find
    auto [it1, inserted1] = m.emplace(1u, 10u);
    assert(inserted1);
    assert(m.size() == 1);
    assert(it1->first == 1u && it1->second == 10u);

    auto it_found = m.find(1u);
    assert(it_found != m.end());
    assert(it_found->second == 10u);
    assert(m.at(1u) == 10u);

    // insert duplicate key
    auto [it2, inserted2] = m.emplace(1u, 20u);
    assert(!inserted2);
    assert(m.size() == 1);
    assert(it2->second == 10u); // emplace 不覆盖

    // operator[]
    m[2u] = 200u;
    assert(m.size() == 2);
    assert(m.at(2u) == 200u);
    assert(m.capacity() == 8);

    m[2u] = 300u;
    assert(m.size() == 2);
    assert(m.at(2u) == 300u);
    assert(m.capacity() == 8);

    UInt32 &ref3 = m[3u];
    assert(ref3 == 0u); // mapped_type{}
    ref3 = 300u;
    assert(m.at(3u) == 300u);
    assert(m.size() == 3);
    assert(m.capacity() == 8);

    // insert or assign
    auto [it3, inserted3] = m.insert_or_assign(2u, 222u);
    assert(!inserted3);
    assert(it3->second == 222u);
    assert(m.at(2u) == 222u);

    auto [it4, inserted4] = m.insert_or_assign(4u, 444u);
    assert(inserted4);
    assert(m.at(4u) == 444u);
    assert(m.size() == 4);
    assert(m.capacity() == 8);

    // at 异常
    bool threw = false;
    try {
        m.at(999u);
    } catch (const std::out_of_range &) { threw = true; }
    assert(threw);
    assert(m.size() == 4);
    assert(m.capacity() == 8);

    // erase by key
    SizeT erased = m.erase(2u);
    assert(erased == 1);
    assert(!m.contains(2u));
    assert(m.size() == 3);
    assert(m.capacity() == 8);

    // clear / 再插入
    m.clear();
    assert(m.empty());
    assert(m.size() == 0);
    assert(m.capacity() == 8);

    auto [it5, inserted5] = m.emplace(42u, 4242u);
    assert(inserted5);
    assert(m.size() == 1);
    assert(m.at(42u) == 4242u);
    assert(m.capacity() == 8);

    m.clear();
    assert(m.size() == 0);
    m.rehash(0);
    assert(m.size() == 0);
    assert(m.capacity() == 0);

    // try_emplace
    auto [te_it1, te_inserted1] = m.try_emplace(5u, 555u);
    assert(te_inserted1);
    assert(te_it1->second == 555u);

    auto [te_it2, te_inserted2] = m.try_emplace(5u, 777u);
    assert(!te_inserted2);
    assert(te_it1->second == 555u);
    assert(m.size() == 1);
    assert(m.capacity() == 8);

    {
        using move_map_t = mcl::flat_hash_map<UInt32, std::unique_ptr<UInt32>>;
        move_map_t mm;
        auto up = std::make_unique<UInt32>(99);
        auto [mit, minserted] = mm.try_emplace(1u, std::move(up));
        assert(minserted);
        assert(up == nullptr);
        assert(*mit->second == 99u);
    }

    // 临界扩容行为
    {
        map_t m2;
        m2.max_load_factor(0.5);
        assert(m2.capacity() == 0);
        m2.try_emplace(10u, 10u);
        assert(m2.capacity() == 8);
        assert(m2.size() == 1);
        m2.try_emplace(20u, 10u);
        m2.try_emplace(30u, 10u);
        m2.try_emplace(40u, 10u);
        assert(m2.size() == 4); // 根据 load factor，还不需要扩容
        assert(m2.capacity() == 8);
        m2.try_emplace(40u, 10u); // 插入重复键，不应该扩容
        assert(m2.capacity() == 8);
    }
}

static void test_insert_range_and_init_list() {
    using map_t = mcl::flat_hash_map<UInt32, UInt32>;
    map_t m;

    std::vector<map_t::value_type> src = {{1u, 10u}, {2u, 20u}, {2u, 200u}};
    m.insert(src.begin(), src.end());
    assert(m.size() == 2);
    assert(m.at(1u) == 10u);
    assert(m.at(2u) == 20u);

    m.insert({{2u, 999u}, {3u, 30u}});
    assert(m.size() == 3);
    assert(m.at(2u) == 20u);
    assert(m.at(3u) == 30u);
}

static void test_shrink_to_fit() {
    using map_t = mcl::flat_hash_map<UInt32, UInt32>;

    map_t m;
    m.emplace(1u, 1u);
    assert(m.capacity() == 8);
    m.clear();
    m.shrink_to_fit();
    assert(m.size() == 0);
    assert(m.capacity() == 0);

    map_t m2;
    m2.max_load_factor(0.5f);
    m2.reserve(64);
    assert(m2.capacity() == 128);

    for (UInt32 i = 0; i < 10; ++i) m2.emplace(i, i * 10u);
    for (UInt32 i = 0; i < 3; ++i) assert(m2.erase(i) == 1);
    assert(m2.size() == 7);

    m2.shrink_to_fit();
    assert(m2.size() == 7);
    assert(m2.capacity() == 16);

    for (UInt32 i = 0; i < 10; ++i) {
        auto it = m2.find(i);
        if (i < 3) assert(it == m2.end());
        else {
            assert(it != m2.end());
            assert(it->second == i * 10u);
        }
    }
}

static void test_high_collision() {
    using map_t = mcl::flat_hash_map<UInt32, UInt32, bad_hash_u32>;
    map_t m;
    constexpr SizeT N = 128;
    for (UInt32 i = 0; i < N; ++i) {
        auto [it, inserted] = m.emplace(i, i * 10u);
        assert(inserted);
        assert(it->first == i);
    }
    assert(m.size() == N);
    assert(m.capacity() == 256);

    for (UInt32 i = 0; i < N; ++i) {
        auto it = m.find(i);
        assert(it != m.end());
        assert(it->second == i * 10u);
    }

    // 删除一半再查找
    for (UInt32 i = 0; i < N; i += 2) {
        SizeT erased = m.erase(i);
        assert(erased == 1);
    }
    assert(m.size() == N / 2);
    assert(m.capacity() == 256);

    for (UInt32 i = 0; i < N; ++i) {
        auto it = m.find(i);
        if (i % 2 == 0) assert(it == m.end());
        else assert(it != m.end());
    }
}

// max load factor / rehash
static void test_load_factor_and_rehash() {
    using map_t = mcl::flat_hash_map<UInt32, UInt32>;
    map_t m;
    m.max_load_factor(0.5f);
    m.reserve(8);
    assert(m.capacity() == 16);

    SizeT max_elems = static_cast<SizeT>(static_cast<float>(m.capacity()) * m.max_load_factor());
    assert(max_elems == 8);

    for (SizeT i = 0; i < max_elems; ++i) {
        auto [it, inserted] = m.emplace(static_cast<UInt32>(i), static_cast<UInt32>(i));
        assert(inserted);
    }
    assert(m.size() == max_elems);
    assert(m.capacity() == 16);

    // 触发 rehash
    auto [it2, inserted2] = m.emplace(static_cast<UInt32>(max_elems), static_cast<UInt32>(max_elems));
    assert(inserted2);
    assert(m.size() == max_elems + 1);
    assert(m.capacity() == 32);
}

static void test_insert_erase_insert_cycle() {
    using map_t = mcl::flat_hash_map<UInt32, UInt32>;

    map_t m;
    m.max_load_factor(0.75f);
    m.reserve(16);

    assert(m.capacity() == 32);

    for (UInt32 i = 0; i < 10; ++i) {
        auto [it, insert] = m.emplace(i, i * 10u);
        assert(insert);
    }
    assert(m.size() == 10);

    for (UInt32 i = 0; i < 5; ++i) {
        SizeT erased = m.erase(i);
        assert(erased == 1);
    }
    assert(m.size() == 5);
    assert(m.capacity() == 32);

    for (UInt32 i = 100; i < 105; ++i) {
        auto [it, inserted] = m.emplace(i, i & 10u);
        assert(inserted);
    }
    assert(m.size() == 10);
    assert(m.capacity() == 32);
}

static void test_iterators() {
    using map_t = mcl::flat_hash_map<std::uint32_t, std::uint32_t>;

    map_t m;

    // 完全遍历
    SizeT count = 0;
    for (auto it = m.begin(); it != m.end(); ++it) { ++count; }
    assert(count == m.size());

    constexpr UInt32 N = 128;
    // erase(it++) 直到清空
    {
        map_t m2;
        for (UInt32 i = 0; i < N; ++i) { m2.emplace(i, i * 10u); }
        auto it = m2.begin();
        while (it != m2.end()) it = m2.erase(it);
        assert(m2.empty());
        assert(m2.size() == 0);
    }

    // 区间 erase
    {
        map_t m3;
        for (UInt32 i = 0; i < N; ++i) { m3.emplace(i, i * 10u); }
        auto first = m3.begin();
        std::advance(first, 10); // 跳十次
        auto last = m3.begin();
        std::advance(last, 40);

        // 记录被删除的 key
        std::vector<UInt32> removed_keys;
        for (auto it = first; it != last; ++it) removed_keys.push_back(it->first);

        auto it_after = m3.erase(first, last);
        (void)it_after;

        for (auto k : removed_keys) assert(m3.find(k) == m3.end());

        SizeT remaining = 0;
        for (auto it = m3.begin(); it != m3.end(); ++it) ++remaining;
        assert(remaining + removed_keys.size() == N);
    }
}

static void test_allocator_behavior() {
    using value_type = std::pair<UInt32, UInt32>;
    using map_t =
        mcl::flat_hash_map<UInt32, UInt32, std::hash<UInt32>, std::equal_to<UInt32>, CountingAllocator<value_type>>;

    AllocStats::reset();
    {
        map_t m; // no allocate
        assert(AllocStats::alloc_count == 0);
        m.reserve(32);
        assert(m.capacity() == 64);
        assert(static_cast<SizeT>(AllocStats::alloc_count) == 64); // 分配了 64 个 pair

        for (UInt32 i = 0; i < 32; ++i) m.emplace(i, i * 10u);

        m.max_load_factor(0.5f);
        assert(m.capacity() == 64);
        m.reserve(128); // rehash，分配 256 个新 pair
        assert(m.capacity() == 256);

        for (UInt32 i = 0; i < 16; ++i) (void)m.erase(i);
    }

    assert(AllocStats::alloc_count == 256 + 64);
    assert(AllocStats::dealloc_count == 256 + 64);
}

int main() {
    std::cout << "[flat_hash_map] basic tests running...\n";

    test_basic_operations();
    std::cout << "  basic_operations OK\n";

    test_insert_range_and_init_list();
    std::cout << "  insert range/init list OK\n";

    test_shrink_to_fit();
    std::cout << "  shrink_to_fit OK\n";

    test_high_collision();
    std::cout << "  high_collision OK\n";

    test_load_factor_and_rehash();
    std::cout << "  load_factor / rehash OK\n";

    test_insert_erase_insert_cycle();
    std::cout << "  insert/erase/insert cycle OK\n";

    test_iterators();
    std::cout << "  iterators OK\n";

    test_allocator_behavior();
    std::cout << "  allocator behavior OK\n";

    std::cout << "All flat_hash_map basic tests passed.\n";
    return 0;
}
