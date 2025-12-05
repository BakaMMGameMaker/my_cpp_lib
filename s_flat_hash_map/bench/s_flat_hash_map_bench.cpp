// my_cpp_lib/s_flat_hash_map/bench/s_flat_hash_map_bench.cpp
#include "s_flat_hash_map.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

// 生成 N 个随机 key 用固定种子
template <typename Key> std::vector<Key> make_random_keys(std::size_t N) {
    std::mt19937_64 rng(123456);
    std::vector<Key> keys;
    keys.reserve(N);
    for (std::size_t i = 0; i < N; ++i) { keys.push_back(static_cast<Key>(rng())); }
    return keys;
}

// 生成 N 个必未命中的 key，偏移一个大常数
template <typename Key> std::vector<Key> make_miss_keys(const std::vector<Key> &base) {
    std::vector<Key> miss;
    miss.reserve(base.size());
    Key offset = static_cast<Key>(0x9e3779b97f4a7c15ull);
    for (auto k : base) { miss.push_back(k + offset); }
    return miss;
}

// reserve(N) 后插入 N 个随机 key，对比插入性能
template <class Map, class Key> static void BM_InsertRandomKeys(benchmark::State &state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto keys = make_random_keys<Key>(N);

    for (auto _ : state) {
        Map m;
        m.reserve(N);

        for (std::size_t i = 0; i < N; ++i) { m.emplace(keys[i], static_cast<std::uint32_t>(i)); }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}

// 查 N 次命中
template <class Map, class Key> static void BM_FindHit(benchmark::State &state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto keys = make_random_keys<Key>(N);

    Map m;
    m.reserve(N);
    for (std::size_t i = 0; i < N; ++i) { m.emplace(keys[i], static_cast<std::uint32_t>(i)); }

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto it = m.find(keys[i]);
            benchmark::DoNotOptimize(it);
        }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}

// 查 N 次未命中
template <class Map, class Key> static void BM_FindMiss(benchmark::State &state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto keys = make_random_keys<Key>(N);

    Map m;
    m.reserve(N);
    for (std::size_t i = 0; i < N; ++i) { m.emplace(keys[i], static_cast<std::uint32_t>(i)); }

    const auto miss_keys = make_miss_keys<Key>(keys);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto it = m.find(miss_keys[i]);
            benchmark::DoNotOptimize(it);
        }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}

// 插 N 后，随机删一半，再插一半新 key
template <class Map, class Key> static void BM_EraseHalfInsertHalf(benchmark::State &state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto keys = make_random_keys<Key>(N);
    const auto new_keys = make_random_keys<Key>(N); // 用作再次插入的新 key

    std::vector<Key> erase_order = keys;
    std::mt19937_64 rng(123456);
    std::shuffle(erase_order.begin(), erase_order.end(), rng);

    for (auto _ : state) {
        state.PauseTiming();
        Map m;
        m.reserve(N);
        for (std::size_t i = 0; i < N; ++i) { m.emplace(keys[i], static_cast<std::uint32_t>(i)); }
        state.ResumeTiming();

        const std::size_t half = N / 2;

        // 随机删一半
        for (std::size_t i = 0; i < half; ++i) { (void)m.erase(erase_order[i]); }

        // 插一半新 key
        for (std::size_t i = 0; i < half; ++i) { m.emplace(new_keys[i], static_cast<std::uint32_t>(i)); }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
}

// ----- 注册 benchmark -----
// 范围可以调，给个 2^10, 2^14, 2^18 这种量级
using std_umap_u32 = std::unordered_map<std::uint32_t, std::uint32_t>;
using std_umap_u64 = std::unordered_map<std::uint64_t, std::uint32_t>;

using flat_map_u32 = mcl::flat_hash_map<std::uint32_t, std::uint32_t>;
using flat_map_u64 = mcl::flat_hash_map<std::uint64_t, std::uint32_t>;

#define REG_BENCH_ALL_SCENARIOS(MapType, KeyType, name_prefix)                                                         \
    BENCHMARK_TEMPLATE(BM_InsertRandomKeys, MapType, KeyType)                                                          \
        ->Name(name_prefix "_insert")                                                                                  \
        ->Arg(1 << 10)                                                                                                 \
        ->Arg(1 << 14)                                                                                                 \
        ->Arg(1 << 18);                                                                                                \
    BENCHMARK_TEMPLATE(BM_FindHit, MapType, KeyType)                                                                   \
        ->Name(name_prefix "_find_hit")                                                                                \
        ->Arg(1 << 10)                                                                                                 \
        ->Arg(1 << 14)                                                                                                 \
        ->Arg(1 << 18);                                                                                                \
    BENCHMARK_TEMPLATE(BM_FindMiss, MapType, KeyType)                                                                  \
        ->Name(name_prefix "_find_miss")                                                                               \
        ->Arg(1 << 10)                                                                                                 \
        ->Arg(1 << 14)                                                                                                 \
        ->Arg(1 << 18);                                                                                                \
    BENCHMARK_TEMPLATE(BM_EraseHalfInsertHalf, MapType, KeyType)                                                       \
        ->Name(name_prefix "_erase_half_insert_half")                                                                  \
        ->Arg(1 << 10)                                                                                                 \
        ->Arg(1 << 14)                                                                                                 \
        ->Arg(1 << 18);

// std::unordered_map
REG_BENCH_ALL_SCENARIOS(std_umap_u32, std::uint32_t, "std_umap_u32")
REG_BENCH_ALL_SCENARIOS(std_umap_u64, std::uint64_t, "std_umap_u64")

// flat_hash_map
REG_BENCH_ALL_SCENARIOS(flat_map_u32, std::uint32_t, "flat_map_u32")
REG_BENCH_ALL_SCENARIOS(flat_map_u64, std::uint64_t, "flat_map_u64")

BENCHMARK_MAIN();
