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

// ----------------- key 分布与通用工具 -----------------

enum class KeyDistribution : std::uint32_t { kSequential = 0, kRandom = 1, kClustered = 2 };

constexpr std::array<std::size_t, 3> kSizes = {1 << 10, 1 << 14, 1 << 18};
constexpr std::array<float, 2> kLoadFactors = {0.75f, 0.875f};

template <class Map> void apply_max_load_factor(Map &map, float max_load_factor) {
    map.max_load_factor(max_load_factor);
}

std::string distribution_name(KeyDistribution dist) {
    switch (dist) {
    case KeyDistribution::kSequential:
        return "seq";
    case KeyDistribution::kRandom:
        return "random";
    case KeyDistribution::kClustered:
        return "cluster";
    }
    return "unknown";
}

std::string suffix_for(KeyDistribution dist, float load_factor) {
    std::ostringstream oss;
    oss << distribution_name(dist) << "_lf" << std::setfill('0') << std::setw(3)
        << static_cast<int>(load_factor * 1000.0f + 0.5f);
    return oss.str();
}

std::string build_label(KeyDistribution dist, float load_factor) {
    std::ostringstream oss;
    oss << "dist=" << distribution_name(dist) << " lf=" << std::fixed << std::setprecision(3) << load_factor;
    return oss.str();
}

// ----------------- 数据集生成：仅整数 -----------------

template <typename Key> std::vector<Key> make_random_keys(std::size_t N, std::uint64_t seed = 123456) {
    static_assert(std::is_integral_v<Key>, "random keys require integral types");
    std::mt19937_64 rng(seed);
    std::vector<Key> keys;
    keys.reserve(N);
    for (std::size_t i = 0; i < N; ++i) { keys.push_back(static_cast<Key>(rng())); }
    return keys;
}

template <typename Key> std::vector<Key> make_sequential_keys(std::size_t N) {
    static_assert(std::is_integral_v<Key>, "sequential keys require integral types");
    std::vector<Key> keys;
    keys.reserve(N);
    for (std::size_t i = 0; i < N; ++i) { keys.push_back(static_cast<Key>(i)); }
    return keys;
}

template <typename Key> std::vector<Key> make_cluster_keys(std::size_t N) {
    static_assert(std::is_unsigned_v<Key>, "clustered keys expect unsigned integral types");
    std::mt19937_64 rng(20240527);
    std::uniform_int_distribution<int> offset_dist(0, 63);
    std::vector<Key> keys;
    keys.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        const auto base = static_cast<Key>(rng() & ~static_cast<std::uint64_t>(0x3Fu)); // 聚焦低位，制造 hash 偏斜
        const auto offset = static_cast<Key>(offset_dist(rng));
        keys.push_back(static_cast<Key>(base + offset));
    }
    return keys;
}

template <typename Key> std::vector<Key> make_keys(KeyDistribution dist, std::size_t N) {
    switch (dist) {
    case KeyDistribution::kSequential:
        return make_sequential_keys<Key>(N);
    case KeyDistribution::kRandom:
        return make_random_keys<Key>(N);
    case KeyDistribution::kClustered:
        return make_cluster_keys<Key>(N);
    }
    return {};
}

template <typename Key> std::vector<Key> make_miss_keys(const std::vector<Key> &base) {
    static_assert(std::is_integral_v<Key>, "miss keys require integral types");
    std::vector<Key> miss;
    miss.reserve(base.size());
    constexpr std::uint64_t offset = 0x9e3779b97f4a7c15ull;
    for (auto key : base) { miss.push_back(static_cast<Key>(key + static_cast<Key>(offset))); }
    return miss;
}

template <class Key> struct IntDataset {
    std::vector<Key> keys;      // 命中键
    std::vector<Key> miss_keys; // 未命中键
    std::vector<Key> new_keys;  // 未来插入的新键（当前没用上，预留扩展）
    std::vector<Key> erase_order;
};

template <class Key> IntDataset<Key> prepare_int_dataset(KeyDistribution dist, std::size_t N) {
    IntDataset<Key> data;
    data.keys = make_keys<Key>(dist, N);
    data.miss_keys = make_miss_keys(data.keys);
    data.new_keys = data.miss_keys;
    data.erase_order = data.keys;
    std::mt19937_64 rng(123456);
    std::shuffle(data.erase_order.begin(), data.erase_order.end(), rng);
    return data;
}

// ----------------- Debug 统计（仅 DEBUG 生效） -----------------

template <class Map>
concept HasDebugStats = requires(const Map &map) { map.get_debug_stats(); };

struct DebugCounters {
    template <class Map> void add(const Map &map) {
#if defined(DEBUG)
        if constexpr (HasDebugStats<Map>) {
            const auto stats = map.get_debug_stats();
            total_avg_probe_ += stats.avg_probe_len;
            if (static_cast<std::size_t>(stats.max_probe_len) > max_probe_len_) {
                max_probe_len_ = static_cast<std::size_t>(stats.max_probe_len);
            }
            total_rehash_ += static_cast<double>(stats.rehash_count);
            total_double_rehash_ += static_cast<double>(stats.double_rehash_count);
            total_size_ += static_cast<double>(stats.size);
            total_capacity_ += static_cast<double>(stats.capacity);
            ++samples_;
        }
#else
        (void)map;
#endif
    }

    void publish(benchmark::State &state, float load_factor, std::string_view label) const {
        state.SetLabel(std::string(label));
        state.counters["load_factor"] = static_cast<double>(load_factor);
#if defined(DEBUG)
        if (samples_ == 0) return;
        const double inv = 1.0 / static_cast<double>(samples_);
        state.counters["probe_avg"] = total_avg_probe_ * inv;
        state.counters["probe_max"] = static_cast<double>(max_probe_len_);
        state.counters["rehash"] = total_rehash_ * inv;
        state.counters["double_rehash"] = total_double_rehash_ * inv;
        state.counters["size"] = total_size_ * inv;
        state.counters["capacity"] = total_capacity_ * inv;
#else
        (void)label;
#endif
    }

#if defined(DEBUG)
private:
    double total_avg_probe_ = 0.0;
    std::size_t max_probe_len_ = 0;
    double total_rehash_ = 0.0;
    double total_double_rehash_ = 0.0;
    double total_size_ = 0.0;
    double total_capacity_ = 0.0;
    std::size_t samples_ = 0;
#endif
};

// ----------------- map / policy 别名 -----------------

using Key = std::uint32_t;
using std_umap_u32 = std::unordered_map<std::uint32_t, std::uint32_t, mcl::detail::FastUInt32Hash>;
using fmu32 = mcl::flat_map_u32<std::uint32_t>;

// 插入唯一键场景：知道不会有重复键，所以禁用查重
struct unique_ret_policy {
    static constexpr bool return_value = true;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = false;
};

struct unique_noret_policy {
    static constexpr bool return_value = false;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = false;
};

// 插入重复键场景：必须检查重复键，使用现成策略
using dup_ret_policy = mcl::default_policy;
using dup_noret_policy = mcl::no_return;

// ----------------- Benchmark：Insert（唯一键） -----------------

static void BM_InsertUnique_std(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        std_umap_u32 map;
        apply_max_load_factor(map, max_load_factor);
        map.reserve(N);

        for (std::size_t i = 0; i < N; ++i) {
            auto res = map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }

        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

static void BM_InsertUnique_fmu32_ret(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        fmu32 map;
        apply_max_load_factor(map, max_load_factor);
        map.reserve(N);

        for (std::size_t i = 0; i < N; ++i) {
            auto res = map.template emplace<unique_ret_policy>(dataset.keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }

        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

static void BM_InsertUnique_fmu32_noret(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        fmu32 map;
        apply_max_load_factor(map, max_load_factor);
        map.reserve(N);

        for (std::size_t i = 0; i < N; ++i) {
            // 返回值为 void，不做任何接收
            map.template emplace<unique_noret_policy>(dataset.keys[i], static_cast<std::uint32_t>(i));
        }

        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

// ----------------- Benchmark：Insert（重复键） -----------------

static void BM_InsertDup_std(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    std_umap_u32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);

    // 先插满，后续全部是重复插入
    for (std::size_t i = 0; i < N; ++i) {
        auto res = map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i));
        benchmark::DoNotOptimize(res);
    }

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto res = map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }
        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

static void BM_InsertDup_fmu32_ret(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    fmu32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);

    // 先插满，后续全部是重复插入
    for (std::size_t i = 0; i < N; ++i) {
        auto res = map.template emplace<dup_ret_policy>(dataset.keys[i], static_cast<std::uint32_t>(i));
        benchmark::DoNotOptimize(res);
    }

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto res = map.template emplace<dup_ret_policy>(dataset.keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }
        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

static void BM_InsertDup_fmu32_noret(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    fmu32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);

    // 先插满，后续全部是重复插入
    for (std::size_t i = 0; i < N; ++i) {
        map.template emplace<dup_noret_policy>(dataset.keys[i], static_cast<std::uint32_t>(i));
    }

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            // 无返回值版本，只做插入尝试
            map.template emplace<dup_noret_policy>(dataset.keys[i], static_cast<std::uint32_t>(i));
        }
        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

// ----------------- Benchmark：Find Hit / Miss -----------------

static void BM_FindHit_std(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    std_umap_u32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        auto res = map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i));
        benchmark::DoNotOptimize(res);
    }

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        // 命中查找，但 std 只有 find
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find(dataset.keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

static void BM_FindHit_fmu32(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    fmu32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i)); // 用默认策略建表即可
    }

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        // 这里 100% 确信 key 存在，用 find_exist
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find_exist(dataset.keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

static void BM_FindMiss_std(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    std_umap_u32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        auto res = map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i));
        benchmark::DoNotOptimize(res);
    }

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        // miss 场景只能老老实实用 find
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find(dataset.miss_keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

static void BM_FindMiss_fmu32(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    fmu32 map;
    apply_max_load_factor(map, max_load_factor);
    map.reserve(N);
    for (std::size_t i = 0; i < N; ++i) { map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i)); }

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        // miss 场景肯定不能用 find_exist，不然你就真的原地死循环 live lock
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find(dataset.miss_keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        benchmark::DoNotOptimize(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

// ----------------- 注册所有基准：bench_fmu32_vs_std -----------------

void register_all_benchmarks() {
    constexpr KeyDistribution kDist = KeyDistribution::kRandom; // 只测随机分布，贴近真实负载

    for (float load_factor : kLoadFactors) {
        const std::string suffix = suffix_for(kDist, load_factor);

        // std::unordered_map<u32> 系列
        auto *std_insert_unique = benchmark::RegisterBenchmark(("std_umap_u32_insert_unique_" + suffix).c_str(),
                                                               &BM_InsertUnique_std, kDist, load_factor);
        auto *std_insert_dup = benchmark::RegisterBenchmark(("std_umap_u32_insert_dup_" + suffix).c_str(),
                                                            &BM_InsertDup_std, kDist, load_factor);
        auto *std_find_hit = benchmark::RegisterBenchmark(("std_umap_u32_find_hit_" + suffix).c_str(), &BM_FindHit_std,
                                                          kDist, load_factor);
        auto *std_find_miss = benchmark::RegisterBenchmark(("std_umap_u32_find_miss_" + suffix).c_str(),
                                                           &BM_FindMiss_std, kDist, load_factor);

        // fmu32：插入唯一键，有返回值 vs 无返回值
        auto *fm_insert_unique_ret = benchmark::RegisterBenchmark(("fmu32_ret_insert_unique_" + suffix).c_str(),
                                                                  &BM_InsertUnique_fmu32_ret, kDist, load_factor);
        auto *fm_insert_unique_noret = benchmark::RegisterBenchmark(("fmu32_noret_insert_unique_" + suffix).c_str(),
                                                                    &BM_InsertUnique_fmu32_noret, kDist, load_factor);

        // fmu32：重复插入，有返回值 vs 无返回值
        auto *fm_insert_dup_ret = benchmark::RegisterBenchmark(("fmu32_ret_insert_dup_" + suffix).c_str(),
                                                               &BM_InsertDup_fmu32_ret, kDist, load_factor);
        auto *fm_insert_dup_noret = benchmark::RegisterBenchmark(("fmu32_noret_insert_dup_" + suffix).c_str(),
                                                                 &BM_InsertDup_fmu32_noret, kDist, load_factor);

        // fmu32：find 场景
        auto *fm_find_hit =
            benchmark::RegisterBenchmark(("fmu32_find_hit_" + suffix).c_str(), &BM_FindHit_fmu32, kDist, load_factor);
        auto *fm_find_miss =
            benchmark::RegisterBenchmark(("fmu32_find_miss_" + suffix).c_str(), &BM_FindMiss_fmu32, kDist, load_factor);

        // 统一设置 N
        for (auto size : kSizes) {
            const auto arg = static_cast<std::int64_t>(size);

            std_insert_unique->Arg(arg);
            std_insert_dup->Arg(arg);
            std_find_hit->Arg(arg);
            std_find_miss->Arg(arg);

            fm_insert_unique_ret->Arg(arg);
            fm_insert_unique_noret->Arg(arg);
            fm_insert_dup_ret->Arg(arg);
            fm_insert_dup_noret->Arg(arg);
            fm_find_hit->Arg(arg);
            fm_find_miss->Arg(arg);
        }
    }
}

const bool registered = [] {
    register_all_benchmarks();
    return true;
}();

} // namespace

BENCHMARK_MAIN();
