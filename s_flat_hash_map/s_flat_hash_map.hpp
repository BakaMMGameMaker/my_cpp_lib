#pragma once

#include "salias.h"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mcl {

namespace detail {
// 定义控制字节 control_t = UInt8，有利于 SIMD，并避免花费过多时间比对 key
// - 0x80(10000000) 表示 EMPTY
// - 0xFE(11111110) 表示 DELETED
// - 0xFF(11111111) 表示 SENTINEL 哨兵值，访问到这里说明真实的 control_t 已经全部访问完毕，避免模运算
// - 最高位为 0 代表槽位被占据，即 FULL 范围为 0x00-0x7F (0-127)，冲突概率 1/128
// 虽然浪费了 10000001-11111110 但是为了极致的性能，这是必须放弃的编码空间
// 低 7 位放哈希高位混淆结果，避免全表线性扫描
using control_t = UInt8; // 控制字节

// inline 允许跨多翻译单元重复定义，适用于 header-only 常量
inline constexpr control_t k_empty = static_cast<control_t>(-128); // 0x80 EMPTY
inline constexpr control_t k_deleted = static_cast<control_t>(-2); // 0xFE DELETED
inline constexpr SizeT k_min_capacity = 8;                         // 逻辑允许最大元素数量 = capacity * max load factor
inline constexpr float k_default_max_load_factor = 0.875f;         // 7/8 减少 Group 全 FULL 的概率 (7%)
// EMPTY 非常重要，特别是查找不存在的 key 的时候，越早遇到 key 就能越早返回 false，避免了对整张表完全查询

// 数值是否为 2 的次幂
inline constexpr bool is_power_of_two(SizeT x) noexcept {
    // 如果是 2 的次幂，那么只有一位 1，x - 1 后会得到 0111...，再做按位与与运算结果为 0
    return x && ((x & (x - 1)) == 0);
}

// >= x 的下一个 2 的次幂
// 666 谁写的代码返回值是 bool 已 fixed
// 不要传 <= 1 的值进来，0 分支
inline constexpr SizeT next_power_of_two(SizeT x) noexcept {
    // if (x <= 1) return 1; // 2^0 = 1
    // 通过扩散最高位，避免循环带来的分支，完全内联，缓存友好，更加高效
    --x; // 防止原本就是 2 的次幂的数字跳转到下一个数
    // 往低位扩散最高位
    x |= x >> 1; // 最高位和第二高位都为 1
    x |= x >> 2; // 前四高位都为 1
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;          // 确保 32 位都为 1
#if SIZE_MAX > 0xFFFFFFFFu // 仅在 64bit 架构，即 size_t = UInt64 时执行
    x |= x >> 32;          // 确保 64 位都为 1
#endif
    return x + 1; // 进位，超过最高位时会溢出变为 0，但不是正常表应该出现的情况
}

// 7-bit hash 指纹
inline constexpr control_t short_hash(SizeT h) noexcept {
    // note: h >> 7 是个可调参数，只要 >=7，不要取低位即可
    UInt8 v = static_cast<UInt8>((h >> 7) & 0x7Fu); // 0x7F = 0111 1111
    return static_cast<control_t>(v);
}

template <typename KeyType>
concept HashKey = std::copy_constructible<KeyType> && std::equality_comparable<KeyType> &&
                  std::is_nothrow_move_constructible_v<KeyType>;

template <typename ValueType>
concept HashValue = std::default_initializable<ValueType> && std::move_constructible<ValueType> &&
                    std::destructible<ValueType> && std::is_nothrow_move_constructible_v<ValueType>;

// Hasher 概念约束
// 哈希表内部需要存一个 hasher 对象，copy map, move map 的时候都要构造新的 hasher
// 要求无抛：Hasher 构造与移动本来就不应该抛出，没有必要为它编写复杂的异常安全保证
template <typename HasherType, typename KeyType>
concept HashFor =
    requires(const HasherType &hasher, const KeyType &key) {
        { hasher(key) } noexcept -> std::same_as<SizeT>;    // hasher 可调用且无抛，必须返回 size_t
    } && std::is_nothrow_move_constructible_v<HasherType>   // 能移动构造且无抛
    && std::is_nothrow_default_constructible_v<HasherType>; // 能默认构造且无抛

// Equal 概念约束
// 哈希表内部需要存储判等器，允许用户自定义同一 KeyType 不同的比较逻辑，而不是仅依赖 KeyType 的 operator==
template <typename KeyEqualType, typename KeyType>
concept EqualFor = requires(const KeyEqualType &e, const KeyType &a, const KeyType &b) {
    { e(a, b) } noexcept -> std::same_as<bool>; // 必须返回 bool 且无抛
} && std::is_nothrow_move_constructible_v<KeyEqualType> && std::is_nothrow_default_constructible_v<KeyEqualType>;
} // namespace detail

template <typename KeyType, typename ValueType, typename HasherType = std::hash<KeyType>,
          typename KeyEqualType = std::equal_to<KeyType>, // 比较器，相当于包装了 lhs == rhs 的 functor
          typename Alloc = std::allocator<std::pair<KeyType, ValueType>>>
    requires detail::HashKey<KeyType> && detail::HashValue<ValueType> && detail::HashFor<HasherType, KeyType> &&
             detail::EqualFor<KeyEqualType, KeyType>
class flat_hash_map {
public:
    // control_t = uint8
    // k empty = 0x80
    // k deleted = 0xfe
    // k min capacity = 8
    // k default max load factor = .875
    // 外部注意事项：为了让 next power of two 0 分支，不要给它传入 <= 1 的值
    using key_type = KeyType;
    using mapped_type = ValueType;
    using value_type = std::pair<KeyType, ValueType>; // it->first = new_key 是用户的锅，不是我的
    using size_type = SizeT;
    using difference_type = std::ptrdiff_t;
    using hasher_type = HasherType;
    using key_equal_type = KeyEqualType;
    using allocator_type = Alloc;

private:
    struct Slot {
        value_type kv;
    };

    using control_t = detail::control_t;
    // 通过 allocator traits 包装 allocator type，提供作用明确的统一的接口
    // 不能直接 = std::allocator<Slot>，这样会无视用户传进来的 allocator
    using slot_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<Slot>;
    using slot_alloc_traits = std::allocator_traits<slot_allocator_type>;

public:
    class const_iterator; // 前向声明

    class iterator {
        friend class flat_hash_map;
        friend class const_iterator;

        using map_type = flat_hash_map;

        map_type *map_ = nullptr;
        size_type index_ = 0;

        // 留这个构造接口给容器用
        iterator(map_type *map, size_type index) noexcept : map_(map), index_(index) {}

        // 默认构造函数中没有调用，需在外部自行调用
        void skip_to_next_occupied() noexcept {
            while (index_ < map_->capacity_) {
                control_t control_byte = map_->controls_[index_];
                if (control_byte != detail::k_empty && control_byte != detail::k_deleted) break;
                ++index_;
            }
        }

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = flat_hash_map::value_type;
        using difference_type = flat_hash_map::difference_type;

        iterator() noexcept = default;

        value_type &operator*() const noexcept { return map_->slots_[index_].kv; }

        value_type *operator->() const noexcept { return std::addressof(map_->slots_[index_].kv); }

        iterator &operator++() noexcept {
            ++index_;
            skip_to_next_occupied(); // 如果 it 没绑定到 map，UB
            return *this;
        }

        iterator operator++(int) noexcept {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const iterator &lhs, const iterator &rhs) noexcept {
            return lhs.map_ == rhs.map_ && lhs.index_ == rhs.index_;
        }

        friend bool operator!=(const iterator &lhs, const iterator &rhs) noexcept { return !(lhs == rhs); }
    }; // iterator

    class const_iterator {
        friend class flat_hash_map;

        using map_type = const flat_hash_map;

        const map_type *map_ = nullptr;
        size_type index_ = 0;

        const_iterator(const map_type *map, size_type index) noexcept : map_(map), index_(index) {}

        void skip_to_next_occupied() noexcept {
            while (index_ < map_->capacity_) {
                control_t control_byte = map_->controls_[index_];
                if (control_byte != detail::k_empty && control_byte != detail::k_deleted) break;
                ++index_;
            }
        }

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = flat_hash_map::value_type;
        using difference_type = flat_hash_map::difference_type;

        const_iterator() noexcept = default;
        // 这里不是拷贝构造函数，用于使用 const iterator 接收 iterator
        const_iterator(const iterator &it) noexcept : map_(it.map_), index_(it.index_) {}

        const value_type &operator*() const noexcept { return map_->slots_[index_].kv; }

        const value_type *operator->() const noexcept { return std::addressof(map_->slots_[index_].kv); }

        const_iterator &operator++() noexcept {
            ++index_;
            skip_to_next_occupied();
            return *this;
        }

        const_iterator operator++(int) noexcept {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const const_iterator &lhs, const const_iterator &rhs) noexcept {
            return lhs.map_ == rhs.map_ && lhs.index_ == rhs.index_;
        }

        friend bool operator!=(const const_iterator &lhs, const const_iterator &rhs) noexcept { return !(lhs == rhs); }
    }; // const_iterator

    using node_type = Slot;

    // 默认构造
    flat_hash_map() noexcept(std::is_nothrow_default_constructible_v<HasherType> &&
                             std::is_nothrow_default_constructible_v<KeyEqualType> &&
                             std::is_nothrow_default_constructible_v<Alloc>)
        : hasher_(), equal_(), alloc_(), slot_alloc_(alloc_) {}

    // 接收初始 size 的构造函数
    explicit flat_hash_map(size_type init_size, const HasherType &hasher = HasherType{},
                           const KeyEqualType &equal = KeyEqualType{}, const Alloc &alloc = Alloc{})
        : hasher_(hasher), equal_(equal), alloc_(alloc), slot_alloc_(alloc_) {
        reserve(init_size);
    }

    flat_hash_map(std::initializer_list<value_type> init, const HasherType &hasher = HasherType{},
                  const KeyEqualType &equal = KeyEqualType{}, const Alloc &alloc = Alloc{})
        : hasher_(hasher), equal_(equal), alloc_(alloc), slot_alloc_(alloc_) {
        reserve(init.size());
        // 注意这里的 value_type 是 std::pair<KeyType, ValueType>
        for (const auto &kv : init) { insert(kv); }
    }

    // 拷贝构造：逻辑拷贝，不会完整照搬布局
    flat_hash_map(const flat_hash_map &other)
        : hasher_(other.hasher_), equal_(other.equal_),
          // 根据 Alloc 的定义决定拷贝行为，默认拷贝可能不合预期，如 freelist 指针，arena 指针等不能照搬
          alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)),
          slot_alloc_(alloc_) {
        if (other.size_ == 0) return; // copy empty map 不分配任何 storage
        // 这里虽然用 other.size_ / max load factor 会省空间，但是相当于未来会多几次
        // rehash，所以不要擅自替用户做主，这是他们自己选择的
        rehash(other.capacity_);
        for (size_type index = 0; index < other.capacity_; ++index) {
            control_t control_byte = other.controls_[index];
            if (control_byte == detail::k_empty || control_byte == detail::k_deleted) continue;
            const value_type &kv = other.slots_[index].kv;
            insert(kv);
        }
    }

    // 拷贝赋值，依然是逻辑拷贝
    flat_hash_map &operator=(const flat_hash_map &other) {
        if (this == &other) return *this;
        clear();
        // propagate_on_container_copy_assignment 赋值时是否用右侧的 alloc 替换自己的 alloc
        // 类似的，还有 on move, on swap
        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
            alloc_ = other.alloc_;
            slot_alloc_ = other.slot_alloc_;
        }
        hasher_ = other.hasher_;
        equal_ = other.equal_;
        if (other.size_ == 0) return *this; // copy empty map
        rehash(other.capacity_);
        for (size_type index = 0; index < other.capacity_; ++index) {
            control_t control_byte = other.controls_[index];
            if (control_byte == detail::k_empty || control_byte == detail::k_deleted) continue;
            const value_type &kv = other.slots_[index].kv;
            insert(kv);
        }
        return *this;
    }

    // 移动构造，布局不变
    flat_hash_map(flat_hash_map &&other) noexcept
        : hasher_(std::move(other.hasher_)), equal_(std::move(other.equal_)), alloc_(std::move(other.alloc_)),
          slot_alloc_(std::move(other.slot_alloc_)), controls_(other.controls_), slots_(other.slots_),
          capacity_(other.capacity_), size_(other.size_), deleted_(other.deleted_),
          max_load_factor_(other.max_load_factor_) {
        other.controls_ = nullptr;
        other.slots_ = nullptr;
        other.capacity_ = 0;
        other.size_ = 0;
        other.deleted_ = 0;
    }

    // 移动赋值本质上是用对方的状态覆盖已经存在的现有的状态，所以也要检查有关 alloc 在移动赋值时的行为
    flat_hash_map &
    operator=(flat_hash_map &&other) noexcept(slot_alloc_traits::propagate_on_container_move_assignment::value ||
                                              slot_alloc_traits::is_always_equal::value) {
        if (this == &other) return *this;
        // 准备原封不动地照搬对方的状态
        if (size_ > 0) destroy_all();
        if (capacity_ > 0) deallocate_storage();
        if constexpr (slot_alloc_traits::propagate_on_container_move_assignment::value) {
            alloc_ = std::move(other.alloc_);
            slot_alloc_ = std::move(other.slot_alloc_);
        }
        hasher_ = std::move(other.hasher_);
        equal_ = std::move(other.equal_);
        controls_ = other.controls_;
        slots_ = other.slots_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        deleted_ = other.deleted_;
        max_load_factor_ = other.max_load_factor_;

        other.controls_ = nullptr;
        other.slots_ = nullptr;
        other.capacity_ = 0;
        other.size_ = 0;
        other.deleted_ = 0;
        return *this;
    }

    ~flat_hash_map() {
        if (size_ > 0) destroy_all();
        if (capacity_ > 0) deallocate_storage();
    }

    allocator_type get_allocator() const noexcept { return alloc_; }

    // properties getter
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type deleted() const noexcept { return deleted_; }
    size_type capacity() const noexcept { return capacity_; }
    float load_factor() const noexcept {
        if (capacity_ == 0) return 0.0f;
        return static_cast<float>(size_) / static_cast<float>(capacity_);
    }
    float max_load_factor() const noexcept { return max_load_factor_; }

    // properties setter
    void max_load_factor(float load_factor) {
        // clamp
        if (load_factor < 0.50f) load_factor = 0.50f;
        if (load_factor > 0.95f) load_factor = 0.95f;
        max_load_factor_ = load_factor;
        // 不主动 rehash
    }

    // 不仅销毁全部 kv，还重置所有槽位为 empty
    void clear() noexcept {
        if (size_ > 0) destroy_all();
        if (controls_)
            for (size_type index = 0; index < capacity_; ++index) controls_[index] = detail::k_empty;
        size_ = 0;
        deleted_ = 0;
    }

    // 预留空间
    void reserve(size_type new_size) {
        float need = static_cast<float>(new_size) / max_load_factor_;
        size_type min_capacity = static_cast<size_type>(std::ceil(need));
        if (min_capacity < detail::k_min_capacity) min_capacity = detail::k_min_capacity;
        // 这里不需要 next power of two，rehash 里会调用
        if (min_capacity <= capacity_) return;
        rehash(min_capacity);
    }

    // 交换
    void
    swap(flat_hash_map &other) noexcept(std::allocator_traits<allocator_type>::propagate_on_container_swap::value ||
                                        std::allocator_traits<allocator_type>::is_always_equal::value) {
        if (this == &other) return;
        using std::swap;
        if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value) {
            swap(alloc_, other.alloc_);
            swap(slot_alloc_, other.slot_alloc_);
        }

        swap(hasher_, other.hasher_);
        swap(equal_, other.equal_);
        swap(control_alloc_, other.control_alloc_);
        swap(controls_, other.controls_);
        swap(slots_, other.slots_);
        swap(capacity_, other.capacity_);
        swap(size_, other.size_);
        swap(deleted_, other.deleted_);
        swap(max_load_factor_, other.max_load_factor_);
    }

    // 接收期望容量并重哈希，无活跃元素且 rehash(0) 时变为空 map
    void rehash(size_type new_capacity) {
        // 注意语义是 capacity 不是 size

        // 原本无 storage
        if (capacity_ == 0) {
            if (new_capacity == 0) return; // 空 map rehash 0
            if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity);
            return;
        }

        // 原本有 storage 但无活跃元素
        if (size_ == 0) {
            deallocate_storage();
            deleted_ = 0;
            if (new_capacity == 0) { // 无活跃元素且 rehash(0) -> 空 map
                controls_ = nullptr;
                slots_ = nullptr;
                capacity_ = 0;
                return;
            }
            if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity); // 内部更新 capacity_ = new_capacity，controls_ 和 slot_
            return;
        }

        // capacity_ > 0 and size_ > 0，不允许缩容

        if (new_capacity <= capacity_) {
            new_capacity = capacity_;
        } else {
            if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
        }

        // 到这里：
        // 第一种可能：capacity = new capacity > 0
        // 第二种可能：new capacity = 2^n > capacity

        // 优化：如果新旧容量一致且没有墓碑，什么也不做
        if (new_capacity == capacity_ && deleted_ == 0) return;

        auto old_controls = controls_;
        auto old_slots = slots_;
        auto old_capacity = capacity_;
        allocate_storage(new_capacity); // 里面重置上面三者，所以需要提前保存

        deleted_ = 0; // rehash 之后不会有任何墓碑

        // 搬迁旧元素
        for (size_type index = 0; index < old_capacity; ++index) {
            control_t control_byte = old_controls[index];
            if (control_byte == detail::k_empty || control_byte == detail::k_deleted) continue;
            value_type &kv = old_slots[index].kv;
            size_type hash_result = hash_key(kv.first);
            control_t short_hash_result = detail::short_hash(hash_result);
            size_type insert_index = find_insert_index_for_rehash(hash_result);
            emplace_at_index(insert_index, short_hash_result, std::move(kv.first), std::move(kv.second));
            std::destroy_at(std::addressof(kv)); // 调用析构函数
        }
        // 下面的是合法的
        control_alloc_.deallocate(old_controls, old_capacity);
        slot_alloc_traits::deallocate(slot_alloc_, old_slots, old_capacity);
    }

    iterator begin() noexcept {
        if (!controls_ || size_ == 0) return iterator(this, capacity_); // !controls_ 包含了 capacity_ == 0
        iterator it(this, 0);
        it.skip_to_next_occupied();
        return it;
    }

    // const map.begin = cbegin
    const_iterator begin() const noexcept { return cbegin(); }

    const_iterator cbegin() const noexcept {
        if (!controls_ || size_ == 0) return const_iterator(this, capacity_);
        const_iterator it(this, 0);
        it.skip_to_next_occupied();
        return it;
    }

    iterator end() noexcept { return iterator(this, capacity_); }
    const_iterator end() const noexcept { return cend(); } // const map.end = cend
    const_iterator cend() const noexcept { return const_iterator(this, capacity_); }

    iterator find(const key_type &key) noexcept {
        if (!controls_ || size_ == 0) return end();
        SizeT hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        size_type index = find_index(key, hash_result, short_hash_result);
        if (index == npos) return end();
        return iterator(this, index);
    }

    const_iterator find(const key_type &key) const noexcept {
        if (!controls_ || size_ == 0) return cend();
        SizeT hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        size_type index = find_index(key, hash_result, short_hash_result);
        if (index == npos) return cend();
        return const_iterator(this, index);
    }

    bool contains(const key_type &key) const noexcept { return find(key) != end(); }

    mapped_type &at(const key_type &key) {
        auto it = find(key);
        if (it == end()) throw std::out_of_range("flat_hash_map::at: key not found");
        return it->second; // *it = pair
    }

    const mapped_type &at(const key_type &key) const {
        auto it = find(key);
        if (it == cend()) throw std::out_of_range("flat_hash_map::at: key not found");
        return it->second;
    }

    // 让 operator[] 行为一致，不存在的 key 自动 emplace 一个默认值
    mapped_type &operator[](const key_type &key) {
        SizeT hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, found] = find_or_prepare_insert(key, hash_result, short_hash_result);
        if (!found) emplace_at_index(index, short_hash_result, key, mapped_type{});
        ++size_;
        return slots_[index].kv.second; // 可直接作为等号左侧内容赋值
    }

    // 右值版本，不是万能引用
    mapped_type &operator[](key_type &&key) {
        SizeT hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, found] = find_or_prepare_insert(key, hash_result, short_hash_result);
        if (!found) emplace_at_index(index, short_hash_result, std::move(key), mapped_type{});
        ++size_;
        return slots_[index].kv.second;
    }

    // 注意 value type 是 pair，不是 ValueType
    std::pair<iterator, bool> insert(const value_type &value) { return emplace(value.first, value.second); }

    // 右值版本
    std::pair<iterator, bool> insert(value_type &&value) {
        SizeT hash_result = hash_key(value.first);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, found] = find_or_prepare_insert(value.first, hash_result, short_hash_result);
        if (found) return {iterator(this, index), false};
        emplace_at_index(index, short_hash_result, std::move(value.first), std::move(value.second));
        ++size_;
        return {iterator(this, index), true};
    }

    // 插入或者赋值，return true = insert, false = assign
    template <typename M>                                // 这里 mapped_type / ValueType 已是固定类型，&&
                                                         // 不是万能引用，需要一个新的模板参数
        requires std::constructible_from<mapped_type, M> // 防止乱传
    std::pair<iterator, bool> insert_or_assign(const key_type &key, M &&mapped) {
        SizeT hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, found] = find_or_prepare_insert(key, hash_result, short_hash_result);
        if (found) {
            slots_[index].kv.second = std::forward<M>(mapped);
            return {iterator(this, index), false}; // assign -> false
        }
        emplace_at_index(index, short_hash_result, key, std::forward<M>(mapped));
        ++size_;
        return {iterator(this, index), true};
    }

    // 返回插入或者已存在 key 的 iterator 以及插入是否成功
    template <typename K, typename M>
        requires(std::constructible_from<key_type, K> && std::constructible_from<mapped_type, M>)
    std::pair<iterator, bool> emplace(K &&key, M &&mapped) {
        SizeT hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, found] = find_or_prepare_insert(key, hash_result, short_hash_result);
        if (found) return {iterator(this, index), false};
        // ^^^ key exists / key not exists vvv
        emplace_at_index(index, short_hash_result, std::forward<K>(key), std::forward<M>(mapped));
        ++size_;
        return {iterator(this, index), true};
    }

    // 继承 stl 签名，返回删除了几个元素
    size_type erase(const key_type &key) {
        if (!controls_ || size_ == 0) return 0;
        SizeT hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        size_type index = find_index(key, hash_result, short_hash_result);
        if (index == npos) return 0;
        erase_at_index(index);
        return 1;
    }

    // 按值传递
    iterator erase(iterator pos) {
        if (pos == end()) return end();
        size_type index = pos.index_;
        erase_at_index(index); // erase 不负责 rehash
        iterator it(this, index);
        it.skip_to_next_occupied();
        return it;
    }

    iterator erase(const_iterator pos) {
        if (pos == end()) return end();
        size_type index = pos.index_;
        erase_at_index(index);
        iterator it(this, index);
        it.skip_to_next_occupied();
        return it;
    }

    // 区间擦除
    iterator erase(const_iterator first, const_iterator last) {
        // 不检查是否来自当前 map，UB 是用户自己的锅
        if (first == last) return iterator(this, first.index_);
        size_type begin_index = first.index_;
        size_type end_index = last.index_;
        if (begin_index > capacity_) begin_index = capacity_;
        if (end_index > capacity_) end_index = capacity_;

        // begin index <= capacity_ vvv
        for (size_type index = begin_index; index < end_index; ++index) {
            control_t control_byte = controls_[index];
            if (control_byte != detail::k_empty && control_byte != detail::k_deleted) erase_at_index(index);
        }

        if (end_index >= capacity_) return end();
        iterator it(this, end_index);
        it.skip_to_next_occupied();
        return it;
    }

private:
    static constexpr size_type npos = static_cast<size_type>(-1);

    hasher_type hasher_{};
    key_equal_type equal_{};
    allocator_type alloc_{};
    slot_allocator_type slot_alloc_{};

    control_t *controls_ = nullptr;
    Slot *slots_ = nullptr;
    size_type capacity_ = 0;
    size_type size_ = 0;
    size_type deleted_ = 0;
    float max_load_factor_ = detail::k_default_max_load_factor;

    using control_allocator_type = std::allocator<control_t>; // 确定由 std::allocator 来，不用 traits 包装
    control_allocator_type control_alloc_{};

    SizeT hash_key(const key_type &key) const noexcept { return hasher_(key); }

    // 仅开空间，更新 capacity_，新地方全设为 empty
    void allocate_storage(size_type capacity) {
        controls_ = control_alloc_.allocate(capacity);
        slots_ = slot_alloc_traits::allocate(slot_alloc_, capacity);
        capacity_ = capacity;
        // 初始化为全 EMPTY
        for (size_type index = 0; index < capacity; ++index) { controls_[index] = detail::k_empty; }
    }

    // 仅释放空间，不重置任何成员，保证 capacity > 0 再调用
    void deallocate_storage() noexcept {
        control_alloc_.deallocate(controls_, capacity_);
        slot_alloc_traits::deallocate(slot_alloc_, slots_, capacity_);
    }

    // 仅转一整个 map 摧毁所有活跃 kv
    void destroy_all() noexcept {
        for (size_type index = 0; index < capacity_; ++index) {
            control_t control_byte = controls_[index];
            if (control_byte == detail::k_empty || control_byte == detail::k_deleted) continue;
            std::destroy_at(std::addressof(slots_[index].kv)); // 不会标记为 deleted
        }
    }

    // 专门为 find_or_prepare_insert 准备的快 rehash 路径，由 deleted > size_ * alpha 触发
    // 确保 capacity >= default min capacity
    void cleanup_rehash() {

        deleted_ = 0; // 清理后一定没有墓碑

        // 原本有 storage 但没有活跃元素
        if (size_ == 0) {
            deallocate_storage();
            allocate_storage(capacity_);
            return;
        }

        // 原本有 storage 也有活跃元素
        auto old_controls = controls_;
        auto old_slots = slots_;
        auto old_capacity = capacity_;
        allocate_storage(capacity_); // 里面重置上面三者，所以需要提前保存

        for (size_type index = 0; index < old_capacity; ++index) {
            control_t control_byte = old_controls[index];
            if (control_byte == detail::k_empty || control_byte == detail::k_deleted) continue;
            value_type &kv = old_slots[index].kv;
            size_type hash_result = hash_key(kv.first);
            control_t short_hash_result = detail::short_hash(hash_result);
            size_type insert_index = find_insert_index_for_rehash(hash_result);
            emplace_at_index(insert_index, short_hash_result, std::move(kv.first), std::move(kv.second));
            std::destroy_at(std::addressof(kv)); // 调用析构函数
        }
        control_alloc_.deallocate(old_controls, old_capacity);
        slot_alloc_traits::deallocate(slot_alloc_, old_slots, old_capacity);
    }

    // 专门为 find_or_prepare_insert 准备的快 rehash 路径，确保 capacity >= k min capacity
    // 因为 used 过多而触发，此处 size 不可能为 0，否则 deleted > size_ * alpha 必然成立，会走上面 cleanup_rehash
    // new_capacity = capacity * 2
    void double_storage() {

        auto old_controls = controls_;
        auto old_slots = slots_;
        auto old_capacity = capacity_;
        allocate_storage(capacity_ * 2); // 里面重置上面三者，所以需要提前保存

        deleted_ = 0; // rehash 之后不会有任何墓碑

        // 搬迁旧元素
        for (size_type index = 0; index < old_capacity; ++index) {
            control_t control_byte = old_controls[index];
            if (control_byte == detail::k_empty || control_byte == detail::k_deleted) continue;
            value_type &kv = old_slots[index].kv;
            size_type hash_result = hash_key(kv.first);
            control_t short_hash_result = detail::short_hash(hash_result);
            size_type insert_index = find_insert_index_for_rehash(hash_result);
            emplace_at_index(insert_index, short_hash_result, std::move(kv.first), std::move(kv.second));
            std::destroy_at(std::addressof(kv)); // 调用析构函数
        }
        control_alloc_.deallocate(old_controls, old_capacity);
        slot_alloc_traits::deallocate(slot_alloc_, old_slots, old_capacity);
    }

    // 返回 key 的 index
    size_type find_index(const key_type &key, SizeT hash_result, control_t short_hash_result) const noexcept {
        if (capacity_ == 0) return npos;
        size_type mask = capacity_ - 1;
        size_type index = static_cast<size_type>(hash_result) & mask; // 探测起点

        for (size_type probe = 0; probe < capacity_; ++probe) { // 最多探测 capacity 步
            control_t control_byte = controls_[index];          // 下面更新 index
            if (control_byte == detail::k_empty) return npos;   // EMPTY = 没找到
            // DELETED 会继续找
            if (control_byte == short_hash_result) {
                const key_type &stored_key = slots_[index].kv.first;
                if (equal_(stored_key, key)) return index;
            }
            index = (index + 1) & mask; // 避免模运算，自动回滚到起点
        }
        return npos; // 探测完整个 map 都没找到
    }

    // rehash 搬迁元素专用，allocate_storage 更新 capacity_ 后使用，别让容量为 0
    size_type find_insert_index_for_rehash(SizeT hash_result) {
        // 暂时不需要 short hash result，别乱传
        size_type mask = capacity_ - 1;
        size_type index = static_cast<size_type>(hash_result) & mask;
        for (size_type probe = 0; probe < capacity_; ++probe) {
            control_t control_byte = controls_[index];
            if (control_byte == detail::k_empty) return index; // 新地方没墓碑，别乱来
            index = (index + 1) & mask;
        }
        throw std::logic_error("flat_hash_map::rehash: no empty slot found");
    }

    // 给定 key，存在则返回其 index，否则返回可插入位置，别找到空位不插，里面提前更新属性了
    template <typename K>
        requires std::constructible_from<key_type, K>
    [[nodiscard]] std::pair<size_type, bool> find_or_prepare_insert(const K &key, SizeT hash_result,
                                                                    control_t short_hash_result) {
        if (capacity_ == 0) allocate_storage(detail::k_min_capacity);
        size_type mask = capacity_ - 1;
        size_type index = static_cast<size_type>(hash_result) & mask;
        size_type first_deleted = npos;

        for (size_type probe = 0; probe < capacity_; ++probe) {
            control_t control_byte = controls_[index];
            if (control_byte == detail::k_empty) { // key 的确不存在，需要占用新位置
                // 遇到过 DELETED，重用墓碑，不 rehash
                if (first_deleted != npos) {
                    --deleted_; // 提前更新
                    return {first_deleted, false};
                }
                // 无墓碑，占用 EMPTY，若墓碑太多，以当前容量重哈希，否则扩容一倍
                size_type used = size_ + deleted_; // 已占用 = size + 墓碑
                // + 1 给新的还未被插入的元素做准备
                if (static_cast<float>(used + 1) > max_load_factor_ * static_cast<float>(capacity_)) {
                    // TODO: deleted > size * alpha
                    // 清理墓碑别放外面，不然没事就重哈希，和傻逼一样
                    if (deleted_ > size_ && size_ > 0) cleanup_rehash();
                    else double_storage();
                    return find_or_prepare_insert(key, hash_result, short_hash_result); // 表的结构已经改变，再走一遍
                }
                return {index, false};
            }
            if (control_byte == detail::k_deleted) {
                if (first_deleted == npos) first_deleted = index;
            } else if (control_byte == short_hash_result) {
                const key_type &stored_key = slots_[index].kv.first;
                if (equal_(stored_key, key)) return {index, true};
            }
            index = (index + 1) & mask;
        }
        throw std::logic_error("no empty slot found"); // 不会到这里
    }

    // 在指定位置上填充指纹和 kv，不会更新 size_
    template <typename K, typename M>
        requires(std::constructible_from<key_type, K> && std::constructible_from<mapped_type, M>)
    void emplace_at_index(size_type index, control_t short_hash_result, K &&key, M &&mapped) {
        std::construct_at(std::addressof(slots_[index].kv),
                          std::piecewise_construct,                    // 分片构造
                          std::forward_as_tuple(std::forward<K>(key)), //
                          std::forward_as_tuple(std::forward<M>(mapped)));
        controls_[index] = short_hash_result;
    }

    void erase_at_index(size_type index) noexcept {
        std::destroy_at(std::addressof(slots_[index].kv));
        controls_[index] = detail::k_deleted;
        --size_;
        ++deleted_;
    }

    friend bool operator==(const flat_hash_map &lhs, const flat_hash_map &rhs) {
        if (&lhs == &rhs) return true;
        if (lhs.size_ != rhs.size_) return false;
        if (lhs.size_ == 0) return true; // l.size == r.size == 0

        for (size_type index = 0; index < lhs.capacity_; ++index) {
            control_t control_byte = lhs.controls_[index];
            if (control_byte == detail::k_empty || control_byte == detail::k_deleted) continue;

            const value_type &kv = lhs.slots_[index].kv;
            auto it = rhs.find(kv.first);
            if (it == rhs.end()) return false;            // 对面没有这个键
            if (!(it->second == kv.second)) return false; // 两个键 value 不等
        }
        return true;
    }

    friend void swap(flat_hash_map &lhs, flat_hash_map &rhs) noexcept(noexcept(lhs.swap(rhs))) { lhs.swap(rhs); }
}; // flat_hash_map
} // namespace mcl
