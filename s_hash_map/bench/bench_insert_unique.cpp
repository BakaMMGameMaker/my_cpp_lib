#include "s_alias.h"
#include "s_bucket_map_u32.hpp"
#include "s_flat_map_u32.hpp"
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

// 只测 random 分布即可
enum class KeyDistribution : std::uint32_t { kRandom = 0 };

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

std::string build_label(float load_factor) {
    std::ostringstream oss;
    oss << "insert_unique lf=" << std::fixed << std::setprecision(3) << load_factor;
    return oss.str();
}

// ----------------- 数据集：唯一键 -----------------

template <typename K> std::vector<K> make_unique_keys(UInt32 N, std::uint64_t seed = 123456) {
    static_assert(std::is_integral_v<K>, "random keys require integral types");
    std::vector<K> keys;
    keys.reserve(N);
    for (UInt32 i = 0; i < N; ++i) {
        keys.push_back(static_cast<K>(i)); // 先唯一
    }
    std::mt19937_64 rng(seed);
    std::shuffle(keys.begin(), keys.end(), rng); // 再打乱
    return keys;
}

// ----------------- 别名与 Policy -----------------

using std_umap_u32 = std::unordered_map<std::uint32_t, std::uint32_t, mcl::detail::FastUInt32Hash>;
using fmu32 = mcl::flat_map_u32<std::uint32_t>;
using bmu32 = mcl::bucket_map_u32<std::uint32_t>;
using bm15u32 = mcl::bucket_map_u32_15keys<std::uint32_t>;

// insert unique，保证无重复键：check_dup = false

// reserve + 禁止 rehash
struct unique_reserve_noret_policy {
    static constexpr bool return_value = false;
    static constexpr bool rehash = false;
    static constexpr bool check_dup = false;
};

// no reserve + 允许 rehash
struct unique_noreserve_noret_policy {
    static constexpr bool return_value = false;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = false;
};

// ----------------- Benchmark 实现 -----------------

// --- std::unordered_map: reserve 版本 ---

static void BM_InsertUnique_std_reserve(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        std_umap_u32 map;
        apply_max_load_factor(map, max_load_factor);
        map.reserve(N); // 尽量避免 rehash

        for (UInt32 i = 0; i < N; ++i) {
            auto res = map.emplace(keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// --- std::unordered_map: no reserve 版本 ---

static void BM_InsertUnique_std_noreserve(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        std_umap_u32 map;
        apply_max_load_factor(map, max_load_factor);
        // 不 reserve，交给 std 自己扩容

        for (UInt32 i = 0; i < N; ++i) {
            auto res = map.emplace(keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

static void BM_InsertUnique_fmu32_reserve_noret(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        fmu32 map;
        apply_max_load_factor(map, max_load_factor);
        map.reserve(N);

        for (UInt32 i = 0; i < N; ++i) {
            map.template emplace<unique_reserve_noret_policy>(keys[i], static_cast<std::uint32_t>(i));
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

static void BM_InsertUnique_fmu32_noreserve_noret(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        fmu32 map;
        apply_max_load_factor(map, max_load_factor);

        for (UInt32 i = 0; i < N; ++i) {
            map.template emplace<unique_noreserve_noret_policy>(keys[i], static_cast<std::uint32_t>(i));
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// --- bmu32: reserve + 停止 rehash，无返回值 ---

static void BM_InsertUnique_bmu32_reserve_noret(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        bmu32 map;
        apply_max_load_factor(map, max_load_factor);
        map.reserve(N);

        for (UInt32 i = 0; i < N; ++i) {
            map.template emplace<unique_reserve_noret_policy>(keys[i], static_cast<std::uint32_t>(i));
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// --- bmu32: no reserve + 允许 rehash，无返回值 ---

static void BM_InsertUnique_bmu32_noreserve_noret(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        bmu32 map;
        apply_max_load_factor(map, max_load_factor);

        for (UInt32 i = 0; i < N; ++i) {
            map.template emplace<unique_noreserve_noret_policy>(keys[i], static_cast<std::uint32_t>(i));
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// --- bm15u32: reserve + 停止 rehash，无返回值 ---

static void BM_InsertUnique_bm15u32_reserve_noret(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        bm15u32 map;
        apply_max_load_factor(map, max_load_factor);
        map.reserve(N);

        for (UInt32 i = 0; i < N; ++i) {
            map.template emplace<unique_reserve_noret_policy>(keys[i], static_cast<std::uint32_t>(i));
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// --- bm15u32: no reserve + 允许 rehash，无返回值 ---

static void BM_InsertUnique_bm15u32_noreserve_noret(benchmark::State &state, float max_load_factor) {
    const UInt32 N = static_cast<UInt32>(state.range(0));
    const auto keys = make_unique_keys<Key>(N);

    const auto label = build_label(max_load_factor);

    for (auto _ : state) {
        bm15u32 map;
        apply_max_load_factor(map, max_load_factor);

        for (UInt32 i = 0; i < N; ++i) {
            map.template emplace<unique_noreserve_noret_policy>(keys[i], static_cast<std::uint32_t>(i));
        }

        benchmark::DoNotOptimize(map);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    state.SetLabel(label);
}

// ----------------- 注册：bench_fmu32_vs_std_insert_unique -----------------

void register_all_benchmarks() {
    for (float load_factor : kLoadFactors) {
        const std::string suffix = suffix_for(load_factor);

        // std::unordered_map
        auto *std_reserve = benchmark::RegisterBenchmark(
            ("std_umap_u32_insert_unique_reserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_std_reserve(st, load_factor); });

        auto *std_noreserve = benchmark::RegisterBenchmark(
            ("std_umap_u32_insert_unique_noreserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_std_noreserve(st, load_factor); });

        // fmu32 (no return value)
        auto *fm_reserve_noret = benchmark::RegisterBenchmark(
            ("fmu32_noret_insert_unique_reserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_fmu32_reserve_noret(st, load_factor); });

        auto *fm_noreserve_noret = benchmark::RegisterBenchmark(
            ("fmu32_noret_insert_unique_noreserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_fmu32_noreserve_noret(st, load_factor); });

        // bmu32 (no return value)
        auto *bm_reserve_noret = benchmark::RegisterBenchmark(
            ("bmu32_noret_insert_unique_reserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_bmu32_reserve_noret(st, load_factor); });

        auto *bm_noreserve_noret = benchmark::RegisterBenchmark(
            ("bmu32_noret_insert_unique_noreserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_bmu32_noreserve_noret(st, load_factor); });

        // bm15u32 (no return value)
        auto *bm15_reserve_noret = benchmark::RegisterBenchmark(
            ("bm15u32_noret_insert_unique_reserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_bm15u32_reserve_noret(st, load_factor); });

        auto *bm15_noreserve_noret = benchmark::RegisterBenchmark(
            ("bm15u32_noret_insert_unique_noreserve_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_InsertUnique_bm15u32_noreserve_noret(st, load_factor); });

        for (auto size : kSizes) {
            const auto arg = static_cast<std::int64_t>(size);

            // std_reserve->Arg(arg);
            // std_noreserve->Arg(arg);

            // fm_reserve_noret->Arg(arg);
            // fm_noreserve_noret->Arg(arg);

            bm_reserve_noret->Arg(arg);
            bm_noreserve_noret->Arg(arg);

            bm15_reserve_noret->Arg(arg);
            bm15_noreserve_noret->Arg(arg);
        }
    }
}

const bool registered = [] {
    register_all_benchmarks();
    return true;
}();

} // namespace

BENCHMARK_MAIN();
