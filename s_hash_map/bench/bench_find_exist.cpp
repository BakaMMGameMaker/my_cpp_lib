#include "s_alias.h"
#include "s_bucket_map_u32.hpp"
#include "s_flat_map_u32.hpp"
#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

using Key = std::uint32_t;

constexpr std::array<UInt32, 3> kSizes = {1u << 10, 1u << 14, 1u << 18};
constexpr std::array<float, 2> kLoadFactors = {0.7f, 0.875f};

template <class Map> void apply_max_load_factor(Map &map, float max_load_factor) {
    map.max_load_factor(max_load_factor);
}

std::string suffix_for(float load_factor) {
    std::ostringstream oss;
    oss << "lf" << std::setfill('0') << std::setw(3) << static_cast<int>(load_factor * 1000.0f + 0.5f);
    return oss.str();
}

std::string build_find_label(float load_factor) {
    std::ostringstream oss;
    oss << "find_hit lf=" << std::fixed << std::setprecision(3) << load_factor;
    return oss.str();
}

// ----------------- 数据集：唯一键 + 打乱（随机分布 & 全部存在） -----------------

template <typename K> std::vector<K> make_unique_keys(UInt32 N, std::uint64_t seed = 123456) {
    static_assert(std::is_integral_v<K>, "random keys require integral types");
    std::vector<K> keys;
    keys.reserve(N);
    for (UInt32 i = 0; i < N; ++i) {
        keys.push_back(static_cast<K>(i)); // 保证唯一
    }
    std::mt19937_64 rng(seed);
    std::shuffle(keys.begin(), keys.end(), rng); // 打乱成 random 分布
    return keys;
}

// ----------------- 别名 -----------------

using std_umap_u32 = std::unordered_map<std::uint32_t, std::uint32_t, mcl::detail::FastUInt32Hash>;
using fmu32 = mcl::flat_map_u32<std::uint32_t>;
using bmu32 = mcl::bucket_map_u32<std::uint32_t>;

// 预留 + 唯一键 + 不返回值 + 不触发 rehash，用于建表
struct unique_reserve_noret_policy {
    static constexpr bool return_value = false;
    static constexpr bool rehash = false;
    static constexpr bool check_dup = false;
};

// ----------------- Benchmark 实现：find hit -----------------
//
// 注意：
//  - 建表步骤放在 benchmark 循环外，不计入测量时间
//  - 只测 find 性能
//  - 查找的 key 全部存在
//  - fmu32 使用 find_exist 快速路径
//
// layout 仍然区分 “带 reserve 建表” 和 “不 reserve 建表”，
// 这样可以观察预留容量对最终探测形态的影响。

// --- std::unordered_map: reserve 建表 + find hit ---

static void BM_FindHit_std_reserve(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));

    // 生成唯一随机键
    const auto keys = make_unique_keys<Key>(N);

    // 预先建表，不计入测量时间
    std_umap_u32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N); // 一次性预留，避免 rehash 改变 layout

    for (UInt32 i = 0; i < N; ++i) { map.emplace(keys[i], static_cast<std::uint32_t>(i)); }

    const auto label = build_find_label(max_load_factor);

    for (auto _ : state) {
        for (UInt32 i = 0; i < N; ++i) {
            auto it = map.find(keys[i]); // 命中查找
            benchmark::DoNotOptimize(it);
        }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// --- fmu32: reserve 建表 + find_exist hit ---

static void BM_FindHit_fmu32_reserve(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));

    const auto keys = make_unique_keys<Key>(N);

    fmu32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N); // 一次性预留容量

    // 建表：唯一键插入（noreturn + no rehash）                          // <- 修改
    for (UInt32 i = 0; i < N; ++i) {                       // <- 修改
        map.template emplace<unique_reserve_noret_policy>( // <- 修改
            keys[i], static_cast<std::uint32_t>(i));       // <- 修改
    } // <- 修改

    const auto label = build_find_label(max_load_factor);

    for (auto _ : state) {
        for (UInt32 i = 0; i < N; ++i) {
            auto it = map.find_exist(keys[i]); // 命中路径专用接口
            benchmark::DoNotOptimize(it);
        }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// --- bmu32: reserve 建表 + find_exist hit ---

static void BM_FindHit_bmu32_reserve(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    bmu32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);

    for (UInt32 i = 0; i < N; ++i) {
        map.template emplace<unique_reserve_noret_policy>(keys[i], static_cast<std::uint32_t>(i));
    }

    const auto label = build_find_label(max_load_factor);

    for (auto _ : state) {
        for (UInt32 i = 0; i < N; ++i) {
            auto it = map.find_exist(keys[i]);
            benchmark::DoNotOptimize(it);
        }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// ----------------- 注册：bench_fmu32_vs_std_find_hit -----------------

void register_all_benchmarks() {
    for (float load_factor : kLoadFactors) {
        const std::string suffix = suffix_for(load_factor);

        // std::unordered_map
        auto *std_reserve = benchmark::RegisterBenchmark(
            ("std_umap_u32_find_hit_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_FindHit_std_reserve(st, load_factor); });

        // fmu32
        auto *fm_reserve =
            benchmark::RegisterBenchmark(("fmu32_find_exist_" + suffix).c_str(), [load_factor](benchmark::State &st) {
                BM_FindHit_fmu32_reserve(st, load_factor);
            });

        // bmu32
        auto *bm_reserve =
            benchmark::RegisterBenchmark(("bmu32_find_exist_" + suffix).c_str(), [load_factor](benchmark::State &st) {
                BM_FindHit_bmu32_reserve(st, load_factor);
            });

        for (auto size : kSizes) {
            const auto arg = static_cast<std::int64_t>(size);

            std_reserve->Arg(arg);
            fm_reserve->Arg(arg);
            bm_reserve->Arg(arg);
        }
    }
}

const bool registered = [] {
    register_all_benchmarks();
    return true;
}();

} // namespace

BENCHMARK_MAIN();
