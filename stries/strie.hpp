#pragma once
#include "salias.h"
#include "stries_children_storage.hpp"
#include "stries_free_list.hpp"
#include "sutils.hpp"
#include <algorithm>
#include <ios>
#include <istream>
#include <limits>
#include <memory_resource>
#include <mutex>
#include <ostream>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// 搜索结果返回类型概念约束
template <typename R>
concept PrefixSearchResult = requires(R r) {
    std::ranges::range<R>;                                    // R 有 begin 和 end，可被 for 遍历
    std::same_as<std::ranges::range_value_t<R>, std::string>; // 遍历 R 时得到的每一项类型都为 string
};

// 单词分数计算可调用对象概念约束
template <typename F>
concept ScoreFunc = requires(F f, std::string_view sv) {
    { f(sv) } -> std::convertible_to<SizeT>;
};

// Tries 树概念约束
template <typename T>
concept Tries = requires(T t, const T ct, std::string_view sv, SizeT limit) {
    { t.push(sv) } -> std::same_as<void>;                  // 添加单词
    { t.erase(sv) } -> std::same_as<bool>;                 // 移除单词
    { ct.size() } -> std::same_as<SizeT>;                  // 单词总数
    { ct.empty() } -> std::same_as<bool>;                  // 树是否为空
    { ct.active_node_count() } -> std::same_as<SizeT>;     // 逻辑活跃的节点数
    { ct.contains(sv) } -> std::same_as<bool>;             // 是否包含某单词
    { ct.contains_starts_with(sv) } -> std::same_as<bool>; // 是否包含以给定 sv 为前缀的单词
    { ct.prefix_search(sv, limit) } -> PrefixSearchResult; // 所有以 sv 为前缀的单词
    {
        ct.prefix_search(sv, [](std::string_view) -> SizeT {}, limit)
    } -> PrefixSearchResult;                                        // 按给定方法排序所有以 sv 为前缀的单词
    { ct.longest_prefix_of(sv) } -> std::same_as<std::string_view>; // 最长前缀匹配结果
};

/*
 * @brief 允许用户提供 memory resource ，指定节点中存储孩子的数据结构，是否复用内存池中死亡的节点
 * @tparam ChildrenStorageType 节点中存储孩子的数据结构
 * @tparam ReuseDeadNodes 是否复用内存池中的死亡节点，启用可以节省内存，但降低缓存命中率
 */
template <TriesChildrenStorage ChildrenStorageType, bool ReuseDeadNodes = false> class STrie {

    using IndexType = UInt32;

    static constexpr IndexType invalid_index = std::numeric_limits<UInt32>::max();
    static constexpr UInt32 k_magic = 0x53545249; // 文件头标识
    static constexpr UInt32 k_version = 1;        // 版本号

    struct Node {
        ChildrenStorageType children;
        // 把 is end 压缩到 UInt32 的最高位上，免除 padding？
        // 没有必要，ChildrenStorageType 已经够大了，而且这么做会导致复杂性和可读性都更加恶劣
        bool is_end = false;

        // emplace_back() 时被使用
        Node() = default;

        // emplace_back(memory_resource) 时被使用
        explicit Node(std::pmr::memory_resource *resource)
            requires std::is_constructible_v<ChildrenStorageType, std::pmr::memory_resource *>
            : children(resource), is_end(false) {}

        void reset() noexcept {
            children.reset();
            is_end = false;
        }
    };

    // TODO：deque 版本，但考虑到 deque 版本缓存比较不友好，暂时不考虑，更倾向于一次性 reserve 足够空间
    std::pmr::vector<Node> node_pool;                   // 节点内存池
    TriesFreeList<IndexType, ReuseDeadNodes> free_list; // 空闲节点列表
    std::pmr::memory_resource *memory_resource;         // 用户提供的内存资源
    SizeT word_count;                                   // 单词数量统计
    mutable std::shared_mutex shared_mutex;             // 读写锁

public:
    // 获取单词总数
    [[nodiscard]] SizeT size() const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return word_count;
    }
    // 树是否为空
    [[nodiscard]] bool empty() const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return word_count == 0;
    }
    // 活跃节点总数（启用节点回收时值有意义）
    [[nodiscard]] SizeT active_node_count() const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return node_pool.size() - free_list.size();
    }

private:
    // 创建新的节点
    IndexType create_node() {
        if constexpr (ReuseDeadNodes) {                   // 要复用死亡节点
            if (!free_list.empty()) {                     // 空闲节点列表不为空
                IndexType reused_index = free_list.pop(); // 弹出末尾的空闲节点对应的索引
                Node &node = node_pool[reused_index];     // 获取对应的空闲节点
                node.reset();                             // 重置空闲节点状态
                return reused_index;                      // 返回空闲接待你对应的索引
            }
        }

        // 如果 ChildrenStorageType 有接收 memory_resource* 的构造函数，那么使用它
        if constexpr (std::is_constructible_v<ChildrenStorageType, std::pmr::memory_resource *>)
            node_pool.emplace_back(memory_resource);
        else node_pool.emplace_back(); // 否则创建默认的节点

        return static_cast<IndexType>(node_pool.size() - 1); // 返回新节点的索引
    }

    // 深度优先搜索收集结果
    void dfs(IndexType node_index, std::string &current_word, std::vector<std::string> &result,
             SizeT limit) const noexcept {
        if (result.size() >= limit) return;       // 如果 result 已经到达 limit 直接返回
        const Node &node = node_pool[node_index]; // 获取当前节点
        if (node.is_end) {                        // 如果到当前节点构成完整单词
            result.push_back(current_word);       // 往 result 中加入当前完整单词
            if (result.size() >= limit) return;   // 再次检查 result 是否到达 limit
        }
        if (node.children.empty()) return; // 如果当前节点没有孩子，直接返回

        node.children.for_each_child([this, &current_word, &result, limit](UChar key, IndexType child_node_index) {
            if (child_node_index == invalid_index) return;      // 防御
            if (result.size() >= limit) return;                 // 到达 limit 返回
            current_word.push_back(static_cast<char>(key));     // 往当前单词中加入当前节点对应字符
            dfs(child_node_index, current_word, result, limit); // 递归 dfs
            current_word.pop_back();                            // 在当前单词弹出当前节点对应字符
        });
    }

    // 深度优先搜索维护 Top-K
    template <ScoreFunc ScoreFuncType, typename HeapType>
    void dfs_with_score(IndexType node_index, std::string &current_word, HeapType &heap, SizeT limit,
                        ScoreFuncType &score_func) const {

        using ScoreType = std::invoke_result_t<ScoreFuncType &, std::string_view>;
        using ScoredWord = std::pair<std::string, ScoreType>;

        const Node &node = node_pool[node_index]; // 当前节点
        if (node.is_end) {
            ScoreType score = score_func(std::string_view(current_word));
            if (heap.size() < limit) {
                heap.push(ScoredWord{current_word, score}); // 还没装满堆
            } else if (!heap.empty() && score > heap.top().second) {
                // 仅在分数高于当前堆顶分数时进堆
                heap.pop();
                heap.push(ScoredWord{current_word, score});
            }
        }

        node.children.for_each_child([&](UChar key, IndexType child_node_index) {
            current_word.push_back(static_cast<char>(key));
            dfs_with_score(child_node_index, current_word, heap, limit, score_func);
            current_word.pop_back();
        });
    }

    // 找到 prefix 中最后一个字符对应的节点，返回其索引
    [[nodiscard]] IndexType find_node_index(std::string_view prefix) const noexcept {
        if (node_pool.empty()) [[unlikely]] // 防御
            return invalid_index;
        IndexType current_node_index = 0; // 根节点索引
        for (char ch : prefix) {
            UChar key = CastUChar(ch);
            const Node &current_node = node_pool[current_node_index];    // 当前节点
            IndexType child_node_index = current_node.children.get(key); // 孩子节点索引
            if (child_node_index == invalid_index) return invalid_index; // 孩子节点不存在，返回 invalid
            current_node_index = child_node_index;                       // 更新当前节点索引
        }
        return current_node_index;
    }

    void push_without_lock(std::string_view word) {
        IndexType current_node_index = 0;
        for (char ch : word) {
            UChar key = CastUChar(ch);
            // 不要使用引用，create node 会导致 reallocation
            IndexType child_node_index = node_pool[current_node_index].children.get(key);
            if (child_node_index == invalid_index) {                               // 如果孩子节点不存在
                child_node_index = create_node();                                  // 创建一个孩子节点
                node_pool[current_node_index].children.set(key, child_node_index); // 更新父节点信息
            }
            current_node_index = child_node_index;
        }
        Node &last_node = node_pool[current_node_index];
        if (last_node.is_end) return; // 插入的单词已经存在
        last_node.is_end = true;
        word_count++;
    }

    void clear_without_lock() {
        node_pool.clear();
        if constexpr (ReuseDeadNodes) free_list.clear();
        word_count = 0; // 清空单词计数
        create_node();  // 重新创建根节点
    }

public:
    // 添加新的单词
    void push(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        push_without_lock(word);
    }

    // 移除给定单词，返回移除是否成功
    bool erase(std::string_view word) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        IndexType current_node_index = 0;                                    // 根节点索引
        std::pmr::vector<std::pair<IndexType, UChar>> path(memory_resource); // 记录删除路径 <父节点索引，孩子键>
        path.reserve(word.size());

        for (char ch : word) {
            UChar key = CastUChar(ch);
            Node &current_node = node_pool[current_node_index];          // 当前节点
            IndexType child_node_index = current_node.children.get(key); // 孩子节点索引
            if (child_node_index == invalid_index) return false;         // 单词不存在，删除失败
            path.emplace_back(current_node_index, key);                  // 新增 <当前节点，孩子键>
            current_node_index = child_node_index;                       // 更新当前节点索引
        }

        Node &last_node = node_pool[current_node_index]; // 单词末尾节点
        if (!last_node.is_end) return false;             // 单词在树中不构成完整单词，删除失败
        last_node.is_end = false;                        // 末尾节点不再为单词末尾节点
        if (!last_node.children.empty()) {               // 末尾节点有孩子，无需进一步删除，删除成功
            word_count--;
            return true;
        }

        // 回溯：把所有非单词末尾且没有孩子的无效节点都删除
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            IndexType parent_node_index = it->first;                          // 父节点索引
            UChar child_key = it->second;                                     // 孩子键
            Node &parent_node = node_pool[parent_node_index];                 // 父节点
            IndexType child_node_index = parent_node.children.get(child_key); // 孩子节点索引
            if (child_node_index == invalid_index) [[unlikely]]               // 防御
                break;
            const Node &child_node = node_pool[child_node_index];           // 孩子节点
            if (child_node.is_end || !child_node.children.empty()) break;   // 为单词末尾或有孩子，无需进一步删除
            parent_node.children.set(child_key, invalid_index);             // 更新父节点，删除孩子节点
            if constexpr (ReuseDeadNodes) free_list.push(child_node_index); // 更新空闲节点列表
        }
        word_count--;
        return true;
    }

    // 查找是否包含给定单词
    [[nodiscard]] bool contains(std::string_view word) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        IndexType node_index = find_node_index(word);
        return node_index != invalid_index && node_pool[node_index].is_end; // 能找到对应节点且对应节点 is end 为真
    }

    // 查找是否包含以给定 prefix 为前缀的单词
    [[nodiscard]] bool contains_starts_with(std::string_view prefix) const noexcept {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        return find_node_index(prefix) != invalid_index;
    }

    // 获取最多 limit 个以 prefix 为前缀的单词
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, SizeT limit = SizeMax) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        if (limit == 0) return {};

        IndexType start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        std::vector<std::string> result;
        result.reserve(limit == SizeMax ? 16 : limit);
        std::string current_word(prefix);
        current_word.reserve(prefix.size() + 32);
        dfs(start_node_index, current_word, result, limit);
        return result;
    }

    // 获取最多 limit 个以 prefix 为前缀的单词，以传入方法计算单词分数，结果由高到低排列
    template <ScoreFunc ScoreFuncType>
    [[nodiscard]] std::vector<std::string> prefix_search(std::string_view prefix, ScoreFuncType &&score_func,
                                                         SizeT limit = SizeMax) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        using NoRefScoreFuncType = std::remove_reference_t<ScoreFuncType>;
        using ScoreType = std::invoke_result_t<NoRefScoreFuncType &, std::string_view>;
        using ScoredWord = std::pair<std::string, ScoreType>;

        if (limit == 0) return {};

        IndexType start_node_index = find_node_index(prefix);
        if (start_node_index == invalid_index) return {};

        // 优先队列得 Top-K
        auto cmp = [](const ScoredWord &lhs, const ScoredWord &rhs) { return lhs.second > rhs.second; }; // 小根堆
        std::priority_queue<ScoredWord, std::vector<ScoredWord>, decltype(cmp)> heap(cmp);

        NoRefScoreFuncType local_score_func = std::forward<ScoreFuncType>(score_func);

        std::string current_word(prefix);
        dfs_with_score(start_node_index, current_word, heap, limit, local_score_func);

        if (heap.empty()) return {};

        std::vector<std::string> result;
        result.reserve(heap.size());
        while (!heap.empty()) {
            result.push_back(std::move(heap.top().first)); // move 应该没什么用，因为 top 返回 const reference
            heap.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    // 获取给定文本的所有前缀中在树中存在且作为完整单词的最长结果
    [[nodiscard]] std::string_view longest_prefix_of(std::string_view text) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);
        IndexType current_node_index = 0; // 根节点索引
        SizeT last_match_pos = 0;         // 最后匹配成功的位置
        bool found = false;
        for (SizeT i = 0; i < text.size(); ++i) {
            UChar key = CastUChar(text[i]);
            const Node &current_node = node_pool[current_node_index];    // 当前节点
            IndexType child_node_index = current_node.children.get(key); // 孩子节点索引
            if (child_node_index == invalid_index) break;                // 孩子节点不存在
            const Node &child_node = node_pool[child_node_index];        // 孩子节点
            current_node_index = child_node_index;
            if (!child_node.is_end) continue;
            last_match_pos = i + 1;
            found = true;
        }
        if (!found) return {};
        return text.substr(0, last_match_pos);
    }

    void clear() noexcept {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);
        clear_without_lock();
    }

    // 将 Trie 序列化到输出流
    // [uint32 magic][uint32 version][uint64 word_count]
    // 重复 word_count 次：[uint64 len][len 字节的字符数据（不含'\0'）]
    void serialize(std::ostream &os) const {
        std::shared_lock<std::shared_mutex> read_lock(shared_mutex);

        // 写入文件头
        const UInt32 magic = k_magic;
        const UInt32 version = k_version;
        os.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        os.write(reinterpret_cast<const char *>(&version), sizeof(version));

        // 收集所有单词
        std::vector<std::string> words;
        words.reserve(word_count); // word count 是逻辑单词数量

        std::string current_word;
        current_word.reserve(32);

        // 从根结点出发进行 dfs，收集所有的单词
        if (!node_pool.empty()) dfs(0, current_word, words, SizeMax);

        const UInt64 count = static_cast<UInt64>(words.size());
        os.write(reinterpret_cast<const char *>(&count), sizeof(count));

        // 写入每一个单词 [长度][字节数据]
        for (const auto &word : words) {
            const UInt64 len = static_cast<UInt64>(word.size());
            os.write(reinterpret_cast<const char *>(&len), sizeof(len));
            if (len != 0) os.write(word.data(), static_cast<std::streamsize>(len));
        }

        if (!os) throw std::runtime_error("STrie::serialize: write failure");
    }

    // 反序列化覆盖当前 Trie 内容
    void deserialize(std::istream &is) {
        std::unique_lock<std::shared_mutex> write_lock(shared_mutex);

        // 读取并检查文件头
        UInt32 magic = 0;
        UInt32 version = 0;
        is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char *>(&version), sizeof(version));
        if (!is) throw std::runtime_error("STrie::deserialize: failed to read header");
        if (magic != k_magic) throw std::runtime_error("STrie::deserialize: invalid magic");
        if (version != k_version) throw std::runtime_error("STrie::deserialize: unsupported version");

        // 读取单词数量
        UInt64 count = 0;
        is.read(reinterpret_cast<char *>(&count), sizeof(count));
        if (!is) throw std::runtime_error("STrie::deserialize: failed to read word count");

        // 清空原内容，重建节点
        clear_without_lock();

        // 逐个读出单词并插入
        std::string word;
        for (UInt64 i = 0; i < count; ++i) {
            UInt64 len = 0;
            is.read(reinterpret_cast<char *>(&len), sizeof(len));
            if (!is) throw std::runtime_error("STrie::deserialize: failed to read word length");
            if (len == 0) continue;
            word.resize(static_cast<SizeT>(len));
            is.read(word.data(), static_cast<std::streamsize>(len));
            if (!is) throw std::runtime_error("STrie::deserialize: failed to read word bytes");
            push_without_lock(word);
        }
    }

    explicit STrie(std::pmr::memory_resource *resource = std::pmr::get_default_resource(),
                   SizeT initial_capacity = 1024)
        : node_pool(resource), free_list(resource), memory_resource(resource), word_count(0), shared_mutex() {
        node_pool.reserve(initial_capacity);
        create_node();
    }
};