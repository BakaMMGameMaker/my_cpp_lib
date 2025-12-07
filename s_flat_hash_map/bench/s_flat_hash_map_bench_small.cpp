#include "s_flat_hash_map.hpp"
#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
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

std::string suffix_for(std::string_view scenario, float load_factor) {
    std::ostringstream oss;
    oss << scenario << "_lf" << std::setfill('0') << std::setw(3) << static_cast<int>(load_factor * 1000.0f + 0.5f);
    return oss.str();
}

std::string build_label(KeyDistribution dist, float load_factor) {
    std::ostringstream oss;
    oss << "dist=" << distribution_name(dist) << " lf=" << std::fixed << std::setprecision(3) << load_factor;
    return oss.str();
}

std::string build_string_label(float load_factor) {
    std::ostringstream oss;
    oss << "short_str lf=" << std::fixed << std::setprecision(3) << load_factor;
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
    std::vector<Key> miss_keys;
    std::vector<Key> new_keys;
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

template <class Map, class Key> void fill_map_with_keys(Map &map, const std::vector<Key> &keys) {
    map.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) { map.emplace(keys[i], static_cast<std::uint32_t>(i)); }
}

template <class Map, class Key>
static void BM_InsertKeys(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);
    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        Map map;
        apply_max_load_factor(map, max_load_factor);
        fill_map_with_keys(map, dataset.keys);
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

template <class Map, class Key>
static void BM_InsertDuplicateKeys(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    Map map;
    apply_max_load_factor(map, max_load_factor);
    fill_map_with_keys(map, dataset.keys); // 先插满，后面全部是重复插入

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto res = map.emplace(dataset.keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

template <class Map, class Key>
static void BM_FindMiss(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);

    Map map;
    apply_max_load_factor(map, max_load_factor);
    fill_map_with_keys(map, dataset.keys); // miss_keys 必定不存在

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find(dataset.miss_keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

template <class Map, class Key>
static void BM_FindHit(benchmark::State &state, KeyDistribution dist, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_int_dataset<Key>(dist, N);
    Map map;
    apply_max_load_factor(map, max_load_factor);
    fill_map_with_keys(map, dataset.keys);

    DebugCounters counters;
    const auto label = build_label(dist, max_load_factor);
    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find(dataset.keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

// ------------- 短字符串 string_view 场景 -------------

std::vector<std::string> make_short_strings(std::size_t N, std::uint64_t seed = 321987) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> len_dist(4, 16);
    std::vector<std::string> out;
    out.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        const auto len = static_cast<std::size_t>(len_dist(rng));
        std::string s(len, 'a');
        std::uint64_t val = rng() ^ (static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull);
        for (std::size_t j = 0; j < len; ++j) {
            val ^= val >> 12;
            val ^= val << 25;
            val ^= val >> 27;
            s[j] = static_cast<char>('a' + (val & 15)); // 只用低 4 bit，保持在可打印范围
            val = val * 0x2545F4914F6CDD1Dull + (j + 1);
        }
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<std::string> make_miss_strings(const std::vector<std::string> &base) {
    std::vector<std::string> miss = base;
    for (auto &s : miss) {
        if (s.empty()) {
            s = "miss";
            continue;
        }
        const char rotated = static_cast<char>('a' + ((s.front() - 'a' + 1) % 26));
        s.front() = rotated;
    }
    return miss;
}

std::vector<std::string_view> make_views(const std::vector<std::string> &storage) {
    std::vector<std::string_view> views;
    views.reserve(storage.size());
    for (const auto &s : storage) { views.emplace_back(s); }
    return views;
}

struct StringDataset {
    std::vector<std::string> storage;
    std::vector<std::string_view> keys;
    std::vector<std::string> miss_storage;
    std::vector<std::string_view> miss_keys;
    std::vector<std::string> new_storage;
    std::vector<std::string_view> new_keys;
    std::vector<std::size_t> erase_order;
};

StringDataset prepare_string_dataset(std::size_t N) {
    StringDataset data;
    data.storage = make_short_strings(N, 321987);
    data.keys = make_views(data.storage);
    data.miss_storage = make_miss_strings(data.storage);
    data.miss_keys = make_views(data.miss_storage);
    data.new_storage = make_short_strings(N, 987654);
    data.new_keys = make_views(data.new_storage);
    data.erase_order.resize(N);
    for (std::size_t i = 0; i < N; ++i) { data.erase_order[i] = i; }
    std::mt19937_64 rng(123456);
    std::shuffle(data.erase_order.begin(), data.erase_order.end(), rng);
    return data;
}

template <class Map, class Views> std::vector<typename Map::key_type> materialize_keys(const Views &views) {
    std::vector<typename Map::key_type> out;
    out.reserve(views.size());
    for (auto v : views) { out.emplace_back(v); }
    return out;
}

template <class Map> struct StringKeysForMap {
    std::vector<typename Map::key_type> keys;
    std::vector<typename Map::key_type> miss_keys;
    std::vector<typename Map::key_type> new_keys;
};

template <class Map> StringKeysForMap<Map> materialize_map_keys(const StringDataset &dataset) {
    StringKeysForMap<Map> out;
    out.keys = materialize_keys<Map>(dataset.keys);
    out.miss_keys = materialize_keys<Map>(dataset.miss_keys);
    out.new_keys = materialize_keys<Map>(dataset.new_keys);
    return out;
}

template <class Map> static void BM_StringInsert(benchmark::State &state, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_string_dataset(N);
    const auto map_keys = materialize_map_keys<Map>(dataset);
    DebugCounters counters;
    const auto label = build_string_label(max_load_factor);

    for (auto _ : state) {
        Map map;
        apply_max_load_factor(map, max_load_factor);
        fill_map_with_keys(map, map_keys.keys);
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

template <class Map> static void BM_StringFindHit(benchmark::State &state, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_string_dataset(N);
    const auto map_keys = materialize_map_keys<Map>(dataset);
    Map map;
    apply_max_load_factor(map, max_load_factor);
    fill_map_with_keys(map, map_keys.keys);

    DebugCounters counters;
    const auto label = build_string_label(max_load_factor);
    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find(map_keys.keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}
template <class Map> static void BM_StringInsertDuplicate(benchmark::State &state, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_string_dataset(N);
    const auto map_keys = materialize_map_keys<Map>(dataset);

    Map map;
    apply_max_load_factor(map, max_load_factor);
    fill_map_with_keys(map, map_keys.keys); // 先插一次，后面全是重复插入

    DebugCounters counters;
    const auto label = build_string_label(max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto res = map.emplace(map_keys.keys[i], static_cast<std::uint32_t>(i));
            benchmark::DoNotOptimize(res);
        }
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

template <class Map> static void BM_StringFindMiss(benchmark::State &state, float max_load_factor) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto dataset = prepare_string_dataset(N);
    const auto map_keys = materialize_map_keys<Map>(dataset);

    Map map;
    apply_max_load_factor(map, max_load_factor);
    fill_map_with_keys(map, map_keys.keys); // miss_keys 肯定不在表里

    DebugCounters counters;
    const auto label = build_string_label(max_load_factor);

    for (auto _ : state) {
        for (std::size_t i = 0; i < N; ++i) {
            auto it = map.find(map_keys.miss_keys[i]);
            benchmark::DoNotOptimize(it);
        }
        state.PauseTiming();
        counters.add(map);
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(N));
    counters.publish(state, max_load_factor, label);
}

// ---------------- 注册所有基准 ----------------

using std_umap_u32 = std::unordered_map<std::uint32_t, std::uint32_t>;
using std_umap_u64 = std::unordered_map<std::uint64_t, std::uint32_t>;
using flat_map_u32 = mcl::flat_hash_map<std::uint32_t, std::uint32_t>;
using flat_map_u64 = mcl::flat_hash_map<std::uint64_t, std::uint32_t>;

using std_umap_string = std::unordered_map<std::string, std::uint32_t>;
using std_umap_sview = std::unordered_map<std::string_view, std::uint32_t>;
using flat_map_sview = mcl::flat_hash_map<std::string_view, std::uint32_t>;

template <class Map, class Key> void register_int_family(const std::string &prefix) {
    // 只测随机分布，最能体现实际表现
    constexpr KeyDistribution kDist = KeyDistribution::kRandom;

    for (float load_factor : kLoadFactors) {
        const std::string suffix = suffix_for(kDist, load_factor);

        auto *insert = benchmark::RegisterBenchmark((prefix + "_insert_" + suffix).c_str(), &BM_InsertKeys<Map, Key>,
                                                    kDist, load_factor);

        auto *find_hit = benchmark::RegisterBenchmark((prefix + "_find_hit_" + suffix).c_str(), &BM_FindHit<Map, Key>,
                                                      kDist, load_factor);
        auto *insert_dup = benchmark::RegisterBenchmark((prefix + "_insert_dup_" + suffix).c_str(),
                                                        &BM_InsertDuplicateKeys<Map, Key>, kDist, load_factor);

        auto *find_miss = benchmark::RegisterBenchmark((prefix + "_find_miss_" + suffix).c_str(),
                                                       &BM_FindMiss<Map, Key>, kDist, load_factor);

        for (auto size : kSizes) {
            const auto arg = static_cast<std::int64_t>(size);
            insert->Arg(arg);
            find_hit->Arg(arg);
            insert_dup->Arg(arg);
            find_miss->Arg(arg);
        }
    }
}

template <class Map> void register_string_family(const std::string &prefix) {
    for (float load_factor : kLoadFactors) {
        const std::string suffix = suffix_for("short", load_factor);

        auto *insert =
            benchmark::RegisterBenchmark((prefix + "_insert_" + suffix).c_str(), [load_factor](benchmark::State &st) {
                BM_StringInsert<Map>(st, load_factor);
            });

        auto *find_hit =
            benchmark::RegisterBenchmark((prefix + "_find_hit_" + suffix).c_str(), [load_factor](benchmark::State &st) {
                BM_StringFindHit<Map>(st, load_factor);
            });

        auto *insert_dup = benchmark::RegisterBenchmark(
            (prefix + "_insert_dup_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_StringInsertDuplicate<Map>(st, load_factor); });

        auto *find_miss = benchmark::RegisterBenchmark(
            (prefix + "_find_miss_" + suffix).c_str(),
            [load_factor](benchmark::State &st) { BM_StringFindMiss<Map>(st, load_factor); });

        for (auto size : kSizes) {
            const auto arg = static_cast<std::int64_t>(size);
            insert->Arg(arg);
            find_hit->Arg(arg);
            insert_dup->Arg(arg);
            find_miss->Arg(arg);
        }
    }
}

void register_all_benchmarks() {
    // 整数：只测 u32 随机分布，std vs flat
    register_int_family<std_umap_u32, std::uint32_t>("std_umap_u32");
    register_int_family<flat_map_u32, std::uint32_t>("flat_map_u32");

    // 短字符串：只测 std::string vs flat_map<std::string_view>
    register_string_family<std_umap_string>("std_umap_string");
    register_string_family<flat_map_sview>("flat_map_sview");
}

const bool registered = [] {
    register_all_benchmarks();
    return true;
}();

} // namespace

BENCHMARK_MAIN();
