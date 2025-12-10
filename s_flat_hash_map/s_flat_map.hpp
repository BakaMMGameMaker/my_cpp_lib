// s_flat_hash_map/s_flat_hash_map.cpp
#pragma once
#include "s_alias.h"
#include "s_detail.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <emmintrin.h>
#include <functional>
#include <immintrin.h>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <mmintrin.h>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vcruntime_new_debug.h>

// #define DEBUG

namespace mcl {

template <typename KeyType, typename ValueType, typename HasherType = std::hash<KeyType>,
          typename KeyEqualType = std::equal_to<KeyType>, // 比较器，相当于包装了 lhs == rhs 的 functor
          typename Alloc = std::allocator<std::pair<KeyType, ValueType>>>
    requires detail::HashKey<KeyType> && detail::HashValue<ValueType> && detail::HashFor<HasherType, KeyType> &&
             detail::EqualFor<KeyEqualType, KeyType>
class flat_hash_map {
public:
    // control_t = uint8
    // k empty = 0x80
    // k min capacity = 8
    // k default max load factor = .75
    // 不要给 next power of two 传入 <= 1 的值
    using key_type = KeyType;
    using mapped_type = ValueType;
    using value_type = std::pair<KeyType, ValueType>;
    using size_type = UInt32;
    using difference_type = std::ptrdiff_t;
    using hasher_type = HasherType;
    using key_equal_type = KeyEqualType;
    using allocator_type = Alloc;

private:
    static constexpr bool k_store_hash = detail::k_store_hash<key_type, hasher_type>;

    struct SlotHashStorage {
        size_type hash;
    };

    struct SlotNoHashStorage {};

    struct Slot : std::conditional_t<k_store_hash, SlotHashStorage, SlotNoHashStorage> {
        value_type kv;
    };

#ifdef DEBUG
    struct debug_stats {
        size_type size;
        size_type capacity;
        SizeT rehash_count;
        SizeT double_rehash_count;
        SizeT max_probe_len;
        double avg_probe_len;
    };
#endif

    using control_t = detail::control_t;
    // 通过 allocator traits 包装 allocator type，提供作用明确的统一的接口
    // 不能直接 = std::allocator<Slot>，这样会无视用户传进来的 allocator
    using slot_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<Slot>;
    using slot_alloc_traits = std::allocator_traits<slot_allocator_type>;

public:
    class const_iterator;

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
                if (control_byte != detail::k_empty) break; // 不再判断 k deleted
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
                if (control_byte != detail::k_empty) break;
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
            if (control_byte == detail::k_empty) continue;
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
            if (control_byte == detail::k_empty) continue;
            const value_type &kv = other.slots_[index].kv;
            insert(kv);
        }
        return *this;
    }

    // 移动构造，布局不变
    flat_hash_map(flat_hash_map &&other) noexcept
        : hasher_(std::move(other.hasher_)), equal_(std::move(other.equal_)), alloc_(std::move(other.alloc_)),
          slot_alloc_(std::move(other.slot_alloc_)), controls_(other.controls_), slots_(other.slots_),
          capacity_(other.capacity_), size_(other.size_), max_load_factor_(other.max_load_factor_),
          use_prefetch_(other.use_prefetch_)
#ifdef DEBUG
          ,
          total_probes_(other.total_probes_), probe_ops_(other.probe_ops_), max_probe_len_(other.max_probe_len_),
          rehash_count_(other.rehash_count_), double_rehash_count_(other.double_rehash_count_)
#endif
    {
        other.controls_ = nullptr;
        other.slots_ = nullptr;
        other.capacity_ = 0;
        other.bucket_mask_ = 0;
        other.size_ = 0;
        other.use_prefetch_ = false;
#ifdef DEBUG
        other.total_probes_ = 0;
        other.probe_ops_ = 0;
        other.max_probe_len_ = 0;
        other.rehash_count_ = 0;
        other.double_rehash_count_ = 0;
#endif
    }

    // 用对方状态覆盖现有状态，要检查有关 alloc 在移动赋值时的行为
    flat_hash_map &operator=(flat_hash_map &&other) noexcept {
        static_assert(
            slot_alloc_traits::is_always_equal::value ||
                slot_alloc_traits::propagate_on_container_move_assignment::value,
            "flat_hash_map<UInt32,...> requires allocator that is always_equal or propagates on move assignment");
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
        bucket_mask_ = other.bucket_mask_;
        size_ = other.size_;
        use_prefetch_ = other.use_prefetch_;
        max_load_factor_ = other.max_load_factor_;
#ifdef DEBUG
        total_probes_ = other.total_probes_;
        probe_ops_ = other.probe_ops_;
        max_probe_len_ = other.max_probe_len_;
        rehash_count_ = other.rehash_count_;
        double_rehash_count_ = other.double_rehash_count_;
#endif

        other.controls_ = nullptr;
        other.slots_ = nullptr;
        other.capacity_ = 0;
        other.bucket_mask_ = 0;
        other.size_ = 0;
        other.use_prefetch_ = false;
#ifdef DEBUG
        other.total_probes_ = 0;
        other.probe_ops_ = 0;
        other.max_probe_len_ = 0;
        other.rehash_count_ = 0;
        other.double_rehash_count_ = 0;
#endif
        return *this;
    }

    ~flat_hash_map() {
        if (size_ > 0) destroy_all();
        if (capacity_ > 0) deallocate_storage();
    }

    allocator_type get_allocator() const noexcept { return alloc_; }
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    float load_factor() const noexcept {
        if (capacity_ == 0) return 0.0f;
        return static_cast<float>(size_) / static_cast<float>(capacity_);
    }
    float max_load_factor() const noexcept { return max_load_factor_; }

    // 设置新的 max load factor
    void max_load_factor(float load_factor) {
        // clamp
        if (load_factor < 0.50f) load_factor = 0.50f;
        if (load_factor > 0.95f) load_factor = 0.95f;
        max_load_factor_ = load_factor;
        // 不主动 rehash
    }

    // 销毁全部 kv，重置所有槽位为 empty
    void clear() noexcept {
        if (size_ > 0) destroy_all();
        if (controls_)
            for (size_type index = 0; index < capacity_; ++index) controls_[index] = detail::k_empty;
        size_ = 0;
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

    // 缩减空间
    void shrink_to_fit() {
        if (size_ == 0) {               // 变为空 map
            if (capacity_ == 0) return; // 本身就是空 map
            deallocate_storage();       // 归还空间
            // 重置成员
            controls_ = nullptr;
            slots_ = nullptr;
            capacity_ = 0;
            bucket_mask_ = 0;
            use_prefetch_ = false;
            return;
        }
        // 有活跃元素
        size_type new_capacity = static_cast<size_type>(std::ceil(static_cast<float>(size_) / max_load_factor_));
        if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
        else new_capacity = detail::next_power_of_two(new_capacity);

        if (new_capacity == capacity_) return;

        auto old_controls = controls_;
        auto old_slots = slots_;
        auto old_capacity = capacity_;
        allocate_storage(new_capacity); // 里面重置上面三者，所以需要提前保存
        move_old_elements(old_controls, old_slots, old_capacity);
    }

    // 交换
    void swap(flat_hash_map &other) noexcept {
        static_assert(std::allocator_traits<allocator_type>::propagate_on_container_swap::value ||
                          std::allocator_traits<allocator_type>::is_always_equal::value,
                      "flat_hash_map requires allocator that is always_equal or propagates on swap");
        if (this == &other) return;
        using std::swap;
        if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value) {
            swap(alloc_, other.alloc_);
            swap(slot_alloc_, other.slot_alloc_);
        }

        swap(hasher_, other.hasher_);
        swap(equal_, other.equal_);
        swap(controls_, other.controls_);
        swap(slots_, other.slots_);
        swap(capacity_, other.capacity_);
        swap(size_, other.size_);
        swap(max_load_factor_, other.max_load_factor_);
        swap(use_prefetch_, other.use_prefetch_);
#ifdef DEBUG
        swap(total_probes_, other.total_probes_);
        swap(probe_ops_, other.probe_ops_);
        swap(max_probe_len_, other.max_probe_len_);
        swap(rehash_count_, other.rehash_count_);
        swap(double_rehash_count_, other.double_rehash_count_);
#endif
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
#ifdef DEBUG
            ++rehash_count_;
#endif
            return;
        }

        // 原本有 storage 但无活跃元素
        if (size_ == 0) {
            deallocate_storage();
            if (new_capacity == 0) { // 无活跃元素且 rehash(0) -> 空 map
                controls_ = nullptr;
                slots_ = nullptr;
                capacity_ = 0;
                bucket_mask_ = 0;
                use_prefetch_ = false;
#ifdef DEBUG
                ++rehash_count_;
#endif
                return;
            }
            if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity); // 内部更新 capacity_ = new_capacity，controls_ 和 slot_
#ifdef DEBUG
            ++rehash_count_;
#endif
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
        if (new_capacity == capacity_) return;

        auto old_controls = controls_;
        auto old_slots = slots_;
        auto old_capacity = capacity_;
        allocate_storage(new_capacity); // 里面重置上面三者，所以需要提前保存
#ifdef DEBUG
        ++rehash_count_;
#endif
        move_old_elements(old_controls, old_slots, old_capacity);
    }

    iterator begin() noexcept {
        if (!controls_ || size_ == 0) return iterator(this, capacity_); // !controls_ 包含了 capacity_ == 0
        iterator it(this, 0);
        it.skip_to_next_occupied();
        return it;
    }

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

    // 返回指向给定键值对的迭代器
    iterator find(const key_type &key) noexcept {
        if (!controls_ || size_ == 0) return end();
        size_type hash_result = hash_key(key);
        if constexpr (k_small_int_key) {
            size_type index = simple_find_int(key, hash_result);
            if (index == npos) return end();
            return iterator(this, index);
        } else {
            control_t short_hash_result = detail::short_hash(hash_result);
            // size_type index = robin_hood_find(key, hash_result, short_hash_result);
            size_type index = simple_find(key, hash_result, short_hash_result);
            if (index == npos) return end();
            return iterator(this, index);
        }
    }

    // 返回指向给定键值对的迭代器
    const_iterator find(const key_type &key) const noexcept {
        if (!controls_ || size_ == 0) return cend();
        size_type hash_result = hash_key(key);
        if constexpr (k_small_int_key) {
            size_type index = simple_find_int(key, hash_result);
            if (index == npos) return cend();
            return const_iterator(this, index);
        } else {
            control_t short_hash_result = detail::short_hash(hash_result);
            size_type index = simple_find(key, hash_result, short_hash_result);
            if (index == npos) return cend();
            return const_iterator(this, index);
        }
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

    // 不存在的 key 自动 emplace 一个默认值
    mapped_type &operator[](const key_type &key) {
        size_type hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        size_type index = find_or_insert_default(key, hash_result, short_hash_result);
        return slots_[index].kv.second;
    }

    // 右值版本，不是万能引用
    mapped_type &operator[](key_type &&key) {
        size_type hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        size_type index = find_or_insert_default(std::move(key), hash_result, short_hash_result);
        return slots_[index].kv.second;
    }

    // 尝试插入给定的键值对，返回指向键值对的迭代器以及是否为新的键值对
    std::pair<iterator, bool> insert(const value_type &kv) { return emplace(kv.first, kv.second); }

    // 尝试插入给定的键值对，返回指向键值对的迭代器以及是否为新的键值对（右值版本）
    std::pair<iterator, bool> insert(value_type &&kv) {
        size_type hash_result = hash_key(kv.first);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, inserted] = find_or_insert_kv(hash_result, short_hash_result, std::move(kv));
        return {iterator(this, index), inserted};
    }

    // 插入新的键值对或者覆盖已有键的值，返回真代表插入，返回假代表覆盖
    template <typename K, typename M>
        requires(std::constructible_from<key_type, K> && std::constructible_from<mapped_type, M>)
    std::pair<iterator, bool> insert_or_assign(K &&key, M &&mapped) {
        size_type hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, inserted] =
            insert_or_assign(hash_result, short_hash_result, std::forward<K>(key), std::forward<M>(mapped));
        return {iterator(this, index), inserted};
    }

    // 接收完整键和完整值，返回指向键值对的迭代器以及是否为新的键值对
    template <typename K, typename M>
        requires(std::constructible_from<key_type, K> && std::constructible_from<mapped_type, M>)
    std::pair<iterator, bool> emplace(K &&key, M &&mapped) {
        size_type hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        value_type kv{std::forward<K>(key), std::forward<M>(mapped)};
        auto [index, inserted] = find_or_insert_kv(hash_result, short_hash_result, std::move(kv));
        return {iterator(this, index), inserted};
    }

    // 接收完整键和值的构造参数，返回指向键值对的迭代器以及是否为新的键值对，若键已存在，不覆盖
    template <typename K, typename... Args>
        requires(std::same_as<std::remove_cvref_t<K>, key_type> && std::constructible_from<mapped_type, Args...>)
    std::pair<iterator, bool> try_emplace(K &&key, Args &&...args) {
        size_type hash_result = hash_key(key);
        control_t short_hash_result = detail::short_hash(hash_result);
        auto [index, inserted] =
            find_or_try_emplace(hash_result, short_hash_result, std::forward<K>(key), std::forward<Args>(args)...);
        return {iterator(this, index), inserted};
    }

    // 提供迭代器范围区间并插入键值对
    template <std::input_iterator InputIt>
        requires std::convertible_to<std::iter_reference_t<InputIt>, value_type>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) insert(*first);
    }

    // 提供包含键值对的初始化列表并插入所有元素
    void insert(std::initializer_list<value_type> init) { insert(init.begin(), init.end()); }

    // 返回删除了几个元素
    size_type erase(const key_type &key) {
        if (!controls_ || size_ == 0) return 0;
        size_type hash_result = hash_key(key);
        size_type index;
        if constexpr (k_small_int_key) {
            index = simple_find_int(key, hash_result);
        } else {
            control_t short_hash_result = detail::short_hash(hash_result);
            index = simple_find(key, hash_result, short_hash_result);
        }
        if (index == npos) return 0;
        erase_at_index(index);
        return 1;
    }

    // 移除迭代器指向的键值对并返回指向下一个键值对的迭代器
    iterator erase(const_iterator pos) {
        if (pos == end()) return end();
        size_type index = pos.index_;
        erase_at_index(index); // 这里不检查 kv 是否合法，但是用户本就不应 erase 一个迭代器多次
        iterator it(this, index);
        it.skip_to_next_occupied();
        return it;
    }

    // TODO：添加 erase 范围迭代器接口，但优先级较低

#ifdef DEBUG
    [[nodiscard]] debug_stats get_debug_stats() const noexcept {
        // clang-format off
        return debug_stats {
            size_,
            capacity_,
            rehash_count_,
            double_rehash_count_,
            max_probe_len_,
            probe_ops_ == 0 ? 0.0 : static_cast<double>(total_probes_) / static_cast<double>(probe_ops_)};
        // clang-format on
    }
#endif

private:
    static constexpr size_type npos = static_cast<size_type>(-1);

    // 小整型支持
    static constexpr bool k_small_int_key =
        std::is_integral_v<key_type> && (sizeof(key_type) == 4 || sizeof(key_type) == 8);
    static constexpr size_type k_prefetch_distance = 64;
    static constexpr size_type k_use_prefetch = k_prefetch_distance * 2;
    static constexpr size_type k_prefetch_mask = k_prefetch_distance - 1;
    static constexpr std::align_val_t k_ctrl_align = std::align_val_t{16}; // controls 数组 16 字节对齐

    hasher_type hasher_{};
    key_equal_type equal_{};
    allocator_type alloc_{};
    slot_allocator_type slot_alloc_{};

    control_t *controls_ = nullptr;
    Slot *slots_ = nullptr;
    size_type capacity_ = 0;
    size_type bucket_mask_ = 0;
    size_type size_ = 0;
    float max_load_factor_ = detail::k_default_max_load_factor;
    bool use_prefetch_ = false;

#ifdef DEBUG
    mutable SizeT total_probes_ = 0;        // 探测总长度
    mutable SizeT probe_ops_ = 0;           // 探测次数
    mutable SizeT max_probe_len_ = 0;       // 历史最大探测长度
    mutable SizeT rehash_count_ = 0;        // rehash 次数
    mutable SizeT double_rehash_count_ = 0; // 扩容次数

    void record_probe(SizeT probe_len) const noexcept {
        total_probes_ += probe_len;
        ++probe_ops_;
        if (probe_len > max_probe_len_) max_probe_len_ = probe_len;
    }
#endif

    size_type hash_key(const key_type &key) const noexcept { return static_cast<size_type>(hasher_(key)); }

    size_type get_slot_hash(const Slot &slot) const noexcept {
        if constexpr (k_store_hash) return slot.hash;
        else return hash_key(slot.kv.first); // for cheap hash
    }

    void set_slot_hash(Slot &slot, size_type hash) noexcept {
        if constexpr (k_store_hash) {
            slot.hash = hash;
        } else {
            (void)slot;
            (void)hash;
        }
    }

    // operator[] 专用单次 probing
    // - 若 key 已存在，返回 index，不构造 mapped_type 对象
    // - 若 key 不存在，否则 robin hood 规则插入 {key, default} 并更新 size_
    template <typename K>
        requires(std::constructible_from<key_type, K>) // 开头已限制 mapped type 可默认构造
    size_type find_or_insert_default(K &&key, size_type hash_result, control_t short_hash_result) {
        if (capacity_ == 0) allocate_storage(detail::k_min_capacity);
        // 可能触发扩容，不得不先看 key 是否存在
        if (static_cast<float>(size_ + 1) > max_load_factor_ * static_cast<float>(capacity_)) {
            // size_type existing = robin_hood_find(key, hash_result, short_hash_result);
            size_type existing = simple_find(key, hash_result, short_hash_result);
            if (existing != npos) return existing;
            double_storage(); // capacity *= 2
        }
        size_type index = static_cast<size_type>(hash_result) & bucket_mask_;
        size_type dist = 0;

        alignas(value_type) UChar cur_kv_storage[sizeof(value_type)];
        value_type *cur_kv = nullptr;

        size_type cur_hash = hash_result;
        control_t cur_ctrl = short_hash_result;
        bool need_check_duplicate = true; // 仍需要查找重复键
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            control_t &ctrl_ref = controls_[index];
            Slot &slot_ref = slots_[index];
            if (ctrl_ref == detail::k_empty) {
                ctrl_ref = cur_ctrl;
                set_slot_hash(slot_ref, cur_hash);
                if (!cur_kv) {
                    std::construct_at(std::addressof(slot_ref.kv), std::piecewise_construct,
                                      std::forward_as_tuple(std::forward<K>(key)), std::forward_as_tuple());
                } else {
                    std::construct_at(std::addressof(slot_ref.kv), std::move(*cur_kv));
                    std::destroy_at(cur_kv);
                }
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            if (ctrl_ref == short_hash_result && need_check_duplicate) { // 大部分指纹不匹配
                if (equal_(slot_ref.kv.first, key)) {
#ifdef DEBUG
                    record_probe(probe_len);
#endif
                    return index;
                }
            }
            // 当前 index kv 信息
            SizeT existing_hash = get_slot_hash(slot_ref);
            size_type existing_home = static_cast<size_type>(existing_hash) & bucket_mask_;
            size_type existing_dist = (index + capacity_ - existing_home) & bucket_mask_;
            if (existing_dist < dist) {
                // 进入这里，说明原始 key 不再表中，后续不必查重
                need_check_duplicate = false;
                if (!cur_kv) {
                    cur_kv = std::construct_at(reinterpret_cast<value_type *>(cur_kv_storage), std::piecewise_construct,
                                               std::forward_as_tuple(std::forward<K>(key)), std::forward_as_tuple());
                }
                value_type tmp_kv = std::move(slot_ref.kv);
                size_type tmp_hash = get_slot_hash(slot_ref);
                control_t tmp_ctrl = ctrl_ref;

                slot_ref.kv = std::move(*cur_kv);
                set_slot_hash(slot_ref, cur_hash);
                ctrl_ref = cur_ctrl;

                *cur_kv = std::move(tmp_kv);
                cur_hash = tmp_hash;
                cur_ctrl = tmp_ctrl;
                dist = existing_dist;
            }
            index = (index + 1) & bucket_mask_;
            ++dist;
        }
        throw;
    }

    // 接收右值 kv，若 key 已经存在，返回 kv 索引 + false，否则插入新 kv 返回 insert index + true，并更新 size_
    std::pair<size_type, bool> find_or_insert_kv(size_type hash_result, control_t short_hash_result, value_type &&kv) {
        if (capacity_ == 0) allocate_storage(detail::k_min_capacity);
        if (static_cast<float>(size_ + 1) > max_load_factor_ * static_cast<float>(capacity_)) {
            // size_type existing = robin_hood_find(kv.first, hash_result, short_hash_result);
            size_type existing = simple_find(kv.first, hash_result, short_hash_result);
            if (existing != npos) return {existing, false};
            double_storage();
        }

        size_type index = static_cast<size_type>(hash_result) & bucket_mask_;
        size_type dist = 0;
        value_type cur_kv = std::move(kv);
        size_type cur_hash = hash_result;
        control_t cur_ctrl = short_hash_result;
        bool need_check_duplicate = true;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            control_t &ctrl_ref = controls_[index];
            Slot &slot_ref = slots_[index];
            if (ctrl_ref == detail::k_empty) { // 找到空位直接插入 kv
                ctrl_ref = cur_ctrl;
                set_slot_hash(slot_ref, cur_hash);
                std::construct_at(std::addressof(slot_ref.kv), std::move(cur_kv));
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return {index, true}; // inserted = true
            }
            if (ctrl_ref == short_hash_result && need_check_duplicate) {
                if (equal_(slot_ref.kv.first, cur_kv.first)) {
#ifdef DEBUG
                    record_probe(probe_len);
#endif
                    return {index, false}; // inserted = false
                }
            }
            // 当前 index kv 信息
            SizeT existing_hash = get_slot_hash(slot_ref);
            size_type existing_home = static_cast<size_type>(existing_hash) & bucket_mask_;
            size_type existing_dist = (index + capacity_ - existing_home) & bucket_mask_;
            if (existing_dist < dist) {
                need_check_duplicate = false;

                value_type tmp_kv = std::move(slot_ref.kv);
                size_type tmp_hash = get_slot_hash(slot_ref);
                control_t tmp_ctrl = ctrl_ref;

                slot_ref.kv = std::move(cur_kv);
                set_slot_hash(slot_ref, cur_hash);
                ctrl_ref = cur_ctrl;

                cur_kv = std::move(tmp_kv);
                cur_hash = tmp_hash;
                cur_ctrl = tmp_ctrl;
                dist = existing_dist;
            }
            index = (index + 1) & bucket_mask_;
            ++dist;
        }
        throw;
    }

    // 专门为 insert or assign 提供的快路径，单次 probing
    // - 若 key 已存在：直接覆盖，返回 {index, false}
    // - 若 key 不存在：按 robin hood 规则插入 {key, mapped}，返回 {index, true}
    template <typename K, typename M>
        requires(std::same_as<std::remove_cvref_t<K>, key_type> && std::convertible_to<M, mapped_type>)
    std::pair<size_type, bool> insert_or_assign(size_type hash_result, control_t short_hash_result, K &&key,
                                                M &&mapped) {

        if (capacity_ == 0) allocate_storage(detail::k_min_capacity);
        if (static_cast<float>(size_ + 1) > max_load_factor_ * static_cast<float>(capacity_)) {
            // size_type existing = robin_hood_find(key, hash_result, short_hash_result);
            size_type existing = simple_find(key, hash_result, short_hash_result);
            if (existing != npos) {
                slots_[existing].kv.second = std::forward<M>(mapped);
                return {existing, false};
            }
            double_storage();
        }

        size_type index = static_cast<size_type>(hash_result) & bucket_mask_;
        size_type dist = 0;

        alignas(value_type) unsigned char cur_kv_storage[sizeof(value_type)];
        value_type *cur_kv = nullptr;

        size_type cur_hash = hash_result;
        control_t cur_ctrl = short_hash_result;
        bool need_check_duplicate = true;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            control_t &ctrl_ref = controls_[index];
            Slot &slot_ref = slots_[index];
            if (ctrl_ref == detail::k_empty) {
                ctrl_ref = cur_ctrl;
                set_slot_hash(slot_ref, cur_hash);
                // 不能 move，可能把用户传入的左值偷走
                if (!cur_kv) {
                    std::construct_at(std::addressof(slot_ref.kv), std::piecewise_construct,
                                      std::forward_as_tuple(std::forward<K>(key)),
                                      std::forward_as_tuple(std::forward<M>(mapped)));
                } else {
                    std::construct_at(std::addressof(slot_ref.kv), std::move(*cur_kv));
                    std::destroy_at(cur_kv);
                }
                // 下面的 move 不会再拷贝一次，因为 value type 中无 const
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                // 此处 cur kv ptr 必然存在
                return {index, true};
            }
            if (ctrl_ref == short_hash_result && need_check_duplicate) {
                if (equal_(slot_ref.kv.first, key)) {
                    slot_ref.kv.second = std::forward<M>(mapped); // 能 move 就 move，不能 move 就拷贝
#ifdef DEBUG
                    record_probe(probe_len);
#endif
                    // 到这里，cur kv 一定为空，要是 cur kv 被构造过，压根不会有重复 key
                    return {index, false}; // 覆盖
                }
            }
            SizeT existing_hash = get_slot_hash(slot_ref);
            size_type existing_home = static_cast<size_type>(existing_hash) & bucket_mask_;
            size_type existing_dist = (index + capacity_ - existing_home) & bucket_mask_;
            if (existing_dist < dist) {
                need_check_duplicate = false;
                if (!cur_kv) {
                    cur_kv = std::construct_at(reinterpret_cast<value_type *>(cur_kv_storage), std::piecewise_construct,
                                               std::forward_as_tuple(std::forward<K>(key)),
                                               std::forward_as_tuple(std::forward<M>(mapped)));
                }
                value_type tmp_kv = std::move(slot_ref.kv);
                size_type tmp_hash = get_slot_hash(slot_ref);
                control_t tmp_ctrl = ctrl_ref;

                slot_ref.kv = std::move(*cur_kv);
                set_slot_hash(slot_ref, cur_hash);
                ctrl_ref = cur_ctrl;

                *cur_kv = std::move(tmp_kv);
                cur_hash = tmp_hash;
                cur_ctrl = tmp_ctrl;
                dist = existing_dist;
            }
            index = (index + 1) & bucket_mask_;
            ++dist;
        }
        throw;
    }

    // 专门为 try emplace 提供的快路径，接收键和 mapped_type 构造参数，单次 probing
    // - 若 key 已存在，返回 key index, inserted = false，不构造 mapped_type 对象
    // - 若 key 不存在，按 robin hood 规则插入，并在真正需要时才构造 value_type 并更新 size_
    template <typename K, typename... Args>
        requires(std::constructible_from<key_type, K> && std::constructible_from<mapped_type, Args...>)
    std::pair<size_type, bool> find_or_try_emplace(size_type hash_result, control_t short_hash_result, K &&key,
                                                   Args &&...args) {
        if (capacity_ == 0) allocate_storage(detail::k_min_capacity);
        if (static_cast<float>(size_ + 1) > max_load_factor_ * static_cast<float>(capacity_)) {
            // size_type existing = robin_hood_find(key, hash_result, short_hash_result);
            size_type existing = simple_find(key, hash_result, short_hash_result);
            if (existing != npos) return {existing, false};
            double_storage();
        }

        size_type index = static_cast<size_type>(hash_result) & bucket_mask_;
        size_type dist = 0;

        alignas(value_type) unsigned char cur_kv_storage[sizeof(value_type)];
        value_type *cur_kv = nullptr;

        size_type cur_hash = hash_result;
        control_t cur_ctrl = short_hash_result;
        bool need_check_duplicate = true;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            Slot &slot_ref = slots_[index];
            control_t &ctrl_ref = controls_[index];
            if (ctrl_ref == detail::k_empty) {
                ctrl_ref = cur_ctrl;
                set_slot_hash(slot_ref, cur_hash);
                if (!cur_kv) {
                    std::construct_at(std::addressof(slots_[index].kv), std::piecewise_construct,
                                      std::forward_as_tuple(std::forward<K>(key)),
                                      std::forward_as_tuple(std::forward<Args>(args)...));
                } else {
                    std::construct_at(std::addressof(slots_[index].kv), std::move(*cur_kv));
                    std::destroy_at(cur_kv);
                }
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return {index, true}; // inserted = true
            }
            if (ctrl_ref == short_hash_result && need_check_duplicate) {
                if (equal_(slot_ref.kv.first, key)) {
#ifdef DEBUG
                    record_probe(probe_len);
#endif
                    return {index, false}; // inserted = false
                }
            }
            SizeT existing_hash = get_slot_hash(slot_ref);
            size_type existing_home = static_cast<size_type>(existing_hash) & bucket_mask_;
            size_type existing_dist = (index + capacity_ - existing_home) & bucket_mask_;
            if (existing_dist < dist) {
                need_check_duplicate = false;
                if (!cur_kv) {
                    cur_kv = std::construct_at(reinterpret_cast<value_type *>(cur_kv_storage), std::piecewise_construct,
                                               std::forward_as_tuple(std::forward<K>(key)),
                                               std::forward_as_tuple(std::forward<Args>(args)...));
                }
                value_type tmp_kv = std::move(slot_ref.kv);
                size_type tmp_hash = get_slot_hash(slot_ref);
                control_t tmp_ctrl = ctrl_ref;

                slot_ref.kv = std::move(*cur_kv);
                set_slot_hash(slot_ref, cur_hash);
                ctrl_ref = cur_ctrl;

                *cur_kv = std::move(tmp_kv);
                cur_hash = tmp_hash;
                cur_ctrl = tmp_ctrl;
                dist = existing_dist;
            }
            index = (index + 1) & bucket_mask_;
            ++dist;
        }
        throw;
    }

    // 仅开空间，更新 capacity_，新地方全设为 empty，尾部填充 sentinel
    void allocate_storage(size_type capacity) {
        controls_ = static_cast<control_t *>(
            ::operator new(sizeof(control_t) * (capacity + detail::k_group_width), k_ctrl_align));
        slots_ = slot_alloc_traits::allocate(slot_alloc_, capacity);
        capacity_ = capacity;
        bucket_mask_ = capacity_ - 1;
        use_prefetch_ = capacity_ >= k_use_prefetch;
        // 初始化为全 EMPTY
        for (size_type index = 0; index < capacity_; ++index) { controls_[index] = detail::k_empty; }
        // 尾部 k group width 个位置填充哨兵值，确保任意起点加载 k group width 字节都不会越界
        for (size_type index = capacity_; index < capacity_ + detail::k_group_width; ++index) {
            controls_[index] = detail::k_sentinel;
        }
    }

    // 仅释放空间，不重置任何成员，保证 capacity > 0 再调用
    void deallocate_storage() noexcept {
        ::operator delete[](controls_, k_ctrl_align);
        slot_alloc_traits::deallocate(slot_alloc_, slots_, capacity_);
    }

    // 仅转一整个 map 摧毁所有活跃 kv，不标记 deleted
    void destroy_all() noexcept {
        for (size_type index = 0; index < capacity_; ++index) {
            control_t control_byte = controls_[index];
            if (control_byte == detail::k_empty) continue;
            std::destroy_at(std::addressof(slots_[index].kv));
        }
    }

    // 只负责搬迁旧位置元素到新位置并归还旧空间，在 allocate_storage 后调用
    void move_old_elements(control_t *old_controls, Slot *old_slots, size_type old_capacity) {
        for (size_type index = 0; index < old_capacity; ++index) {
            control_t control_byte = old_controls[index];
            if (control_byte == detail::k_empty) continue;
            value_type &kv = old_slots[index].kv;
            // size_type hash_result = hash_key(kv.first); // 注意，需要测试究竟是再算一次更快，还是取 Slot 中值更快
            size_type hash_result = get_slot_hash(old_slots[index]);
            // active slot 下，short hash result = control byte
            size_type insert_index = find_insert_index_for_rehash(hash_result);
            emplace_at_index(insert_index, hash_result, control_byte, std::move(kv.first), std::move(kv.second));
            std::destroy_at(std::addressof(kv)); // 调用析构函数
        }
        ::operator delete[](old_controls, k_ctrl_align);
        slot_alloc_traits::deallocate(slot_alloc_, old_slots, old_capacity);
    }

    // 专门为 find_or_prepare_insert 准备的快 rehash 路径，确保 capacity >= default min capacity
    // 因为 used 过多而触发，所以此处 size 不可能为 0
    // new_capacity = capacity * 2
    void double_storage() {
        auto old_controls = controls_;
        auto old_slots = slots_;
        auto old_capacity = capacity_;
        allocate_storage(capacity_ * 2);
#ifdef DEBUG
        ++double_rehash_count_;
        ++rehash_count_;
#endif
        move_old_elements(old_controls, old_slots, old_capacity);
    }

    // u32 / u64 等小整型 key 专用查找路径：
    // 不用 short hash，不用 SIMD，只做线性探测 + 直接 key 比较
    // 回归 control byte + SIMD 和 prefetch 都开性能倒车
    size_type simple_find_int(const key_type &key, size_type hash_result) const noexcept {
        if (capacity_ == 0) return npos;
        size_type index = static_cast<size_type>(hash_result) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            // 预取？性能下降，不要再考虑
            // #if defined(__GNUC__) || defined(__clang__)
            //             if constexpr (k_prefetch_distance > 0) {
            //                 if (use_prefetch_ && (index & k_prefetch_mask) == 0) {
            //                     size_type pf = (index + k_prefetch_distance) & bucket_mask_;
            //                     __builtin_prefetch(controls_ + pf);
            //                     __builtin_prefetch(std::addressof(slots_[pf].kv));
            //                 }
            //             }
            // #endif
            control_t control_byte = controls_[index];
            // EMPTY：不存在
            if (control_byte == detail::k_empty) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return npos;
            }
            // 命中 key，直接返回
            if (equal_(slots_[index].kv.first, key)) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            index = (index + 1) & bucket_mask_;
        }
        return npos;
    }

    // 小表，不走 SIMD
    size_type simple_find_small(const key_type &key, size_type hash_result,
                                control_t short_hash_result) const noexcept {
        size_type index = static_cast<size_type>(hash_result) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            control_t control_byte = controls_[index];
            if (control_byte == detail::k_empty) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return npos;
            }
            if (control_byte == short_hash_result && equal_(slots_[index].kv.first, key)) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            index = (index + 1) & bucket_mask_;
        }
        return npos;
    }

    // 提供哈希值和控制字节，返回 key 的 index，不存在则返回 npos，不利用早停机制
    // 还没 umap 一半快，而且 avx2，early exist 等手段全部负收益
    size_type simple_find(const key_type &key, size_type hash_result, control_t short_hash_result) const noexcept {
        if (capacity_ == 0) return npos;

        // capacity < 64 暴力线性扫描
        if (capacity_ < 64) return simple_find_small(key, hash_result, short_hash_result);

        size_type home_bucket = static_cast<size_type>(hash_result) & bucket_mask_; // 探测起点
        size_type first_offset =
            home_bucket & (static_cast<size_type>(detail::k_group_width) - 1); // 起点相对组头位置偏移
        size_type base_bucket = home_bucket - first_offset;                    // 起点组头索引

        const __m128i short_hash_target = _mm_set1_epi8(static_cast<char>(short_hash_result));

#ifdef DEBUG
        SizeT buckets_visited = 0;
#endif
        // circle 1
        {
            const control_t *group_ctrl = controls_ + base_bucket;
            const __m128i ctrl = _mm_load_si128(reinterpret_cast<const __m128i *>(group_ctrl));
            UInt32 group_bits = static_cast<UInt32>(_mm_movemask_epi8(ctrl)); // 收集所有最高位为一个 bitmask
            UInt32 offset_mask = ~UInt32{0} << first_offset;
            UInt32 match_mask = static_cast<UInt32>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl, short_hash_target)));
            UInt32 valid_mask = match_mask & offset_mask; // 屏蔽低位 offset 个 buckets
            while (valid_mask) {
                unsigned bit = static_cast<unsigned>(std::countr_zero(valid_mask)); // 第一个指纹匹配的位置
                size_type index = base_bucket + static_cast<size_type>(bit);
                if (equal_(slots_[index].kv.first, key)) {
#ifdef DEBUG
                    SizeT visited = buckets_visited + (static_cast<SizeT>(bit) - static_cast<SizeT>(first_offset) + 1);
                    record_probe(visited);
#endif
                    return index;
                }
                valid_mask &= (valid_mask - 1);
            }
            UInt32 empty_after = group_bits & offset_mask;
            if (empty_after) {
#ifdef DEBUG
                unsigned bit = static_cast<unsigned>(std::countr_zero(empty_after));
                SizeT visited = buckets_visited + (static_cast<SizeT>(bit) - static_cast<SizeT>(first_offset) + 1);
                record_probe(visited);
#endif
                return npos;
            }
        }
#ifdef DEBUG
        // 整个 first group 从 first_offset 扫到末尾
        buckets_visited += static_cast<SizeT>(detail::k_group_width - first_offset);
#endif
        // 后续组别
        for (;;) {
            base_bucket = (base_bucket + static_cast<size_type>(detail::k_group_width)) & bucket_mask_; // 组头下标
            const control_t *group_ctrl = controls_ + base_bucket; // 指向组头的指针
            const __m128i ctrl = _mm_load_si128(reinterpret_cast<const __m128i *>(group_ctrl));
            UInt32 group_bits = static_cast<UInt32>(_mm_movemask_epi8(ctrl));
            UInt32 match_mask = static_cast<UInt32>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl, short_hash_target)));
            while (match_mask) { // 有任一指纹匹配
                // 低位有多少连续 0，相当于第一个匹配的位置，unsigned 与 countr_zero 搭配
                unsigned bit = static_cast<unsigned>(std::countr_zero(match_mask));
                size_type index = base_bucket + static_cast<size_type>(bit); // 不用取模，范围一定合法
                if (equal_(slots_[index].kv.first, key)) {
#ifdef DEBUG
                    SizeT visited = buckets_visited + static_cast<SizeT>(bit) + 1;
                    record_probe(visited);
#endif
                    return index;
                }
                match_mask &= (match_mask - 1); // 清理最低位的 1
            }
            // 到这里说明当前组没有相同 key，如果 key 存在，至少在下一组
            // 然而如果当前组有 empty 位置，说明 key 不存在
            // UInt32 empty_mask = group_bits
            if (group_bits) { // 有 empty bucket
#ifdef DEBUG
                unsigned bit = static_cast<unsigned>(std::countr_zero(group_bits));
                SizeT visited = buckets_visited + static_cast<SizeT>(bit) + 1;
                record_probe(visited);
#endif
                return npos;
            }
            // 当前组没有 key 也无 empty
#ifdef DEBUG
            // 整组 16 个 bucket 都走完了
            buckets_visited += static_cast<SizeT>(detail::k_group_width);
#endif
        }
        return npos;
    }

    // 提供哈希值和控制字节，返回 key 的 index，不存在则返回 npos
    size_type robin_hood_find(const key_type &key, size_type hash_result, control_t short_hash_result) const noexcept {
        // 除了仅找 key 是否存在的环节别用，节省性能
        if (capacity_ == 0) return npos;
        size_type home = static_cast<size_type>(hash_result) & bucket_mask_; // 探测起点
        size_type index = home;
        size_type dist = 0; // 当前正在处理的 key 相对自己 home 的探测距离
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) { // 最多探测 capacity 步
#ifdef DEBUG
            ++probe_len;
#endif
            control_t control_byte = controls_[index];
            if (control_byte == detail::k_empty) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return npos; // EMPTY = 没找到
            }
            if (control_byte == short_hash_result && equal_(slots_[index].kv.first, key)) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            // TODO: 如果前后两个 hash 相等，说明 home 一样
            // 此时 existing dist 就是 + 1 关系，不需要从头反推
            // 但是会多出很多分支，所以优先级很低
            SizeT existing_hash = get_slot_hash(slots_[index]);
            size_type existing_home = static_cast<size_type>(existing_hash) & bucket_mask_;
            size_type existing_dist = (index + capacity_ - existing_home) & bucket_mask_;
            if (existing_dist < dist) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return npos;
            }
            index = (index + 1) & bucket_mask_; // 避免模运算，自动回滚到起点
            ++dist;
#ifdef DEBUG
            // 不可能到达这里
            if (dist > capacity_) throw std::logic_error("bad map");
#endif
        }
#ifdef DEBUG
        record_probe(probe_len);
#endif
        return npos; // map 中无 key
    }

    // rehash 搬迁元素专用，allocate_storage 更新 capacity_ 后使用，确保 capacity_ > 0 再调用
    // 不会破坏 robin hood probing 探测链，由于不再有墓碑，每个 cluster 都是多个 FULL 紧挨在一起
    // 具体来讲，线性搬迁确保每个元素都在它 home 的右手边，也不会让多个 cluster 混一起，因此安全
    size_type find_insert_index_for_rehash(size_type hash_result) {
        size_type index = static_cast<size_type>(hash_result) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (size_type probe = 0; probe < capacity_; ++probe) {
#ifdef DEBUG
            ++probe_len;
#endif
            control_t control_byte = controls_[index];
            if (control_byte == detail::k_empty) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            index = (index + 1) & bucket_mask_;
        }
        throw std::logic_error("flat_hash_map::rehash: no empty slot found");
    }

    // 在指定位置上填充哈希值，控制字节，并构造 kv，不更新 size_
    template <typename K, typename M>
        requires(std::constructible_from<key_type, K> && std::constructible_from<mapped_type, M>)
    void emplace_at_index(size_type index, size_type hash_result, control_t short_hash_result, K &&key, M &&mapped) {
        std::construct_at(std::addressof(slots_[index].kv),
                          std::piecewise_construct,                    // 分片构造
                          std::forward_as_tuple(std::forward<K>(key)), //
                          std::forward_as_tuple(std::forward<M>(mapped)));
        set_slot_hash(slots_[index], hash_result);
        controls_[index] = short_hash_result;
    }

    // 调用 index 处 kv 析构，标记为墓碑，更新 size_ 和 deleted_，无 rehash 行为
    void erase_at_index(size_type index) noexcept {
        // 逻辑：删完后当前 cluster 剩余元素左移一格，遇到 empty 或者新 cluster 停止
        size_type cur = index;

        for (;;) {
            size_type next = (cur + 1) & bucket_mask_;
            control_t control_byte = controls_[next];
            if (control_byte == detail::k_empty) break;
            size_type next_hash = get_slot_hash(slots_[next]);
            size_type home = static_cast<size_type>(next_hash) & bucket_mask_;
            size_type dist_from_home = (next + capacity_ - home) & bucket_mask_;
            // dist from home = 0 说明是 next cluster，不能左移破坏 probing 链
            if (dist_from_home == 0) break; // index 为 next 的 kv 刚好在自己家里
            controls_[cur] = control_byte;
            slots_[cur].kv = std::move(slots_[next].kv);
            set_slot_hash(slots_[cur], next_hash);
            cur = next;
        }
        std::destroy_at(std::addressof(slots_[cur].kv));
        controls_[cur] = detail::k_empty;
        --size_;
    }

    friend bool operator==(const flat_hash_map &lhs, const flat_hash_map &rhs) {
        if (&lhs == &rhs) return true;
        if (lhs.size_ != rhs.size_) return false;
        if (lhs.size_ == 0) return true; // l.size == r.size == 0

        for (size_type index = 0; index < lhs.capacity_; ++index) {
            control_t control_byte = lhs.controls_[index];
            if (control_byte == detail::k_empty) continue;
            const value_type &kv = lhs.slots_[index].kv;
            // auto it = rhs.robin_hood_find(kv.first);
            auto it = rhs.find(kv.first);
            if (it == rhs.end()) return false;            // 对面没有这个键
            if (!(it->second == kv.second)) return false; // 两个键 value 不等
        }
        return true;
    }

    friend void swap(flat_hash_map &lhs, flat_hash_map &rhs) noexcept(noexcept(lhs.swap(rhs))) { lhs.swap(rhs); }
}; // flat_hash_map
} // namespace mcl
