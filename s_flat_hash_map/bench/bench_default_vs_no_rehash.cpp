#include "s_flat_map_u32.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

// key 分布
enum class KeyDistribution : std::uint32_t { kSequential = 0, kRandom = 1, kClustered = 2 };

constexpr std::array<std::size_t, 2> kSizes = {1 << 14, 1 << 18};
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

template <typename Key> std::vector<Key> make_random_keys(std::size_t N, std::uint64_t seed = 123456) {
    static_assert(std::is_integral_v<Key>, "random keys require integral types");
    std::mt19937_64 rng(seed);
    std::vector<Key> keys;
    keys.reserve(N);
    for (std::size_t i = 0; i < N; ++i) { keys.push_back(static_cast<Key>(rng())); }
    return keys;
}

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

template <class Key> struct IntDataset {
    std::vector<Key> keys;
};

template <class Key> IntDataset<Key> prepare_int_dataset(KeyDistribution dist, std::size_t N) {
    IntDataset<Key> data;
    switch (dist) {
    case KeyDistribution::kRandom:
        data.keys = make_random_keys<Key>(N);
        break;
    case KeyDistribution::kSequential:
    case KeyDistribution::kClustered:
        // 当前只用随机分布就够了，其他分布有需要再开
        data.keys = make_random_keys<Key>(N);
        break;
    }
    return data;
}

// ---------------- emplace / emplace_no_rehash 策略 ----------------

template <bool UseNoRehash, class Map>
inline void bench_emplace(Map &map, const typename Map::key_type &key, std::uint32_t value) {
    if constexpr (UseNoRehash) {
        auto res = map.template emplace<mcl::no_rehash>(key, value);
        benchmark::DoNotOptimize(res);
    } else {
        auto res = map.emplace(key, value);
        benchmark::DoNotOptimize(res);
    }
}

template <bool UseNoRehash, class Map, class Key> void fill_map_with_keys(Map &map, const std::vector<Key> &keys) {
    map.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        bench_emplace<UseNoRehash>(map, keys[i], static_cast<std::uint32_t>(i));
    }
}

// ---------------- 基准本体：insert & insert_dup ----------------

template <bool UseNoRehash, class Map, class Key>
static void BM_InsertKeys(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);
    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        Map map;
        apply_max_load_factor(map, max_load_factor);
        fill_map_with_keys<UseNoRehash>(map, dataset.keys);
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

template <bool UseNoRehash, class Map, class Key>
static void BM_InsertDuplicateKeys(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    Map map;
    apply_max_load_factor(map, max_load_factor);
    fill_map_with_keys<UseNoRehash>(map, dataset.keys); // 先插满

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            bench_emplace<UseNoRehash>(map, dataset.keys[i], static_cast<std::uint32_t>(i));
        }
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

// ---------------- 注册：只测 random insert & insert_dup ----------------

using flat_map_u32 = mcl::flat_hash_map_u32<std::uint32_t>;

template <bool UseNoRehash, class Map, class Key> void register_int_family(const std::string &prefix) {
    // 只测随机分布，最能体现实际表现
    constexpr KeyDistribution kDist = KeyDistribution::kRandom;

    for (float load_factor : kLoadFactors) {
        const std::string suffix = suffix_for(kDist, load_factor);

        auto *insert = benchmark::RegisterBenchmark((prefix + "_insert_" + suffix).c_str(),
                                                    &BM_InsertKeys<UseNoRehash, Map, Key>, kDist, load_factor);

        auto *insert_dup =
            benchmark::RegisterBenchmark((prefix + "_insert_dup_" + suffix).c_str(),
                                         &BM_InsertDuplicateKeys<UseNoRehash, Map, Key>, kDist, load_factor);

        for (auto size : kSizes) {
            const auto arg = static_cast<std::int64_t>(size);
            insert->Arg(arg);
            insert_dup->Arg(arg);
        }
    }
}

void register_all_benchmarks() {
    // checked 路径：正常 emplace
    register_int_family<false, flat_map_u32, std::uint32_t>("flat_map_u32_emplace");

    // unchecked 路径：emplace_no_rehash
    register_int_family<true, flat_map_u32, std::uint32_t>("flat_map_u32_emplace_no_rehash");
}

const bool registered = [] {
    register_all_benchmarks();
    return true;
}();

} // namespace

BENCHMARK_MAIN();
