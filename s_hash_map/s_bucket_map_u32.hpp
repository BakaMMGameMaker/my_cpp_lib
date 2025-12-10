#pragma once
#include "s_alias.h"
#include "s_detail.h"
#include "s_hash_map_policy.hpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace mcl {
template <typename MappedType, typename HasherType = detail::FastUInt32Hash,
          typename Alloc = std::allocator<std::pair<UInt32, MappedType>>>
class bucket_map_u32 {
public:
    using key_type = UInt32;
    using mapped_type = MappedType;
    using value_type = std::pair<const key_type, mapped_type>;
    using size_type = UInt32;
    using difference_type = std::ptrdiff_t;
    using hasher_type = HasherType;
    using key_equal_type = std::equal_to<key_type>;
    using allocator_type = Alloc;

private:
    static constexpr key_type k_unoccupied = std::numeric_limits<UInt32>::max();
    static constexpr size_type k_bucket_width = 16;
    static constexpr size_type k_bucket_width_mask = k_bucket_width - 1;
    static constexpr size_type k_min_capacity = k_bucket_width;
    static constexpr bool is_trivially_destructible_mapped = std::is_trivially_destructible_v<mapped_type>;

    struct Bucket {
        size_type size;
        key_type keys[k_bucket_width];
    };

    static_assert(std::is_trivially_copyable_v<Bucket>, "Bucket must be trivially copyable (size + keys)");

    using bucket_type = Bucket;
    using bucket_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<Bucket>;
    using bucket_alloc_traits = std::allocator_traits<bucket_allocator_type>;
    using mapped_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<mapped_type>;
    using mapped_alloc_traits = std::allocator_traits<mapped_allocator_type>;

#ifdef DEBUG
    struct debug_stats {
        size_type size;
        size_type capacity;
        size_type bucket_count;
        SizeT rehash_count;
        SizeT double_rehash_count;
        SizeT max_bucket_size;
        double avg_bucket_fill;
    };
#endif

    struct value_reference {
        key_type &first;
        mapped_type &second;
        operator value_type() const noexcept { return value_type{first, second}; }
    };

    struct const_value_reference {
        const key_type &first;
        const mapped_type &second;
        operator value_type() const noexcept { return value_type(first, second); }
    };

    struct value_pointer {
        key_type &first;
        mapped_type &second;
        value_pointer *operator->() noexcept { return this; }
        const value_pointer *operator->() const noexcept { return this; }
    };

    struct const_value_pointer {
        const key_type &first;
        const mapped_type &second;
        const const_value_pointer *operator->() const noexcept { return this; }
    };

public:
    class const_iterator;
    class iterator {
        friend class bucket_map_u32;
        friend class const_iterator;

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = bucket_map_u32::difference_type;
        using reference = bucket_map_u32::value_reference;
        using pointer = bucket_map_u32::value_pointer;

    private:
        using map_type = bucket_map_u32;

        map_type *map_ = nullptr;
        bucket_type *buckets_ = nullptr;
        mapped_type *mapped_values_ = nullptr;

        size_type bucket_index_ = 0;
        size_type slot_index_ = 0;

        iterator(map_type *map, size_type bucket_index, size_type slot_index) noexcept
            : map_(map), buckets_(map->buckets_), mapped_values_(map->mapped_values_), bucket_index_(bucket_index),
              slot_index_(slot_index) {}

        void skip_to_next_occupied() noexcept {
            while (bucket_index_ < map_->bucket_count_) {
                if (slot_index_ < buckets_[bucket_index_].size) return;
                // slot index >= bucket.size，到下一个桶的开头
                ++bucket_index_;
                slot_index_ = 0;
            }
        }

    public:
        iterator() noexcept = default;

        reference operator*() const {
            return reference{buckets_[bucket_index_].keys[slot_index_],
                             mapped_values_[bucket_index_ * k_bucket_width + slot_index_]};
        }

        pointer operator->() const {
            return pointer{buckets_[bucket_index_].keys[slot_index_],
                           mapped_values_[bucket_index_ * k_bucket_width + slot_index_]};
        }

        iterator &operator++() {
            ++slot_index_; // 如果到 16，必定触发 slot index >= bucket.size 分支
            skip_to_next_occupied();
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const iterator &lhs, const iterator &rhs) noexcept {
            return lhs.map_ == rhs.map_ && lhs.bucket_index_ == rhs.bucket_index_ && lhs.slot_index_ == rhs.slot_index_;
        }

        friend bool operator!=(const iterator &lhs, const iterator &rhs) noexcept { return !(lhs == rhs); }
    };

    class const_iterator {
        friend class bucket_map_u32;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const bucket_map_u32::value_type;
        using difference_type = bucket_map_u32::difference_type;
        using reference = bucket_map_u32::const_value_reference;
        using pointer = bucket_map_u32::const_value_pointer;

    private:
        using map_type = const bucket_map_u32;

        const map_type *map_ = nullptr;
        const bucket_type *buckets_ = nullptr;
        const mapped_type *mapped_values_ = nullptr;

        size_type bucket_index_ = 0;
        size_type slot_index_ = 0;

        const_iterator(const map_type *map, size_type bucket_index, size_type slot_index) noexcept
            : map_(map), buckets_(map_->buckets_), mapped_values_(map_->mapped_values_), bucket_index_(bucket_index),
              slot_index_(slot_index) {}

        void skip_to_next_occupied() noexcept {
            while (bucket_index_ < map_->bucket_count_) {
                if (slot_index_ < buckets_[bucket_index_].size) return;
                // slot index >= bucket.size，到下一个桶的开头
                ++bucket_index_;
                slot_index_ = 0;
            }
        }

    public:
        const_iterator() noexcept = default;

        const_iterator(const iterator &it) noexcept
            : map_(it.map_), buckets_(it.buckets_), mapped_values_(it.mapped_values_), bucket_index_(it.bucket_index_),
              slot_index_(it.slot_index_) {}

        reference operator*() const {
            return reference{buckets_[bucket_index_].keys[slot_index_],
                             mapped_values_[bucket_index_ * k_bucket_width + slot_index_]};
        }

        pointer operator->() const {
            return pointer{buckets_[bucket_index_].keys[slot_index_],
                           mapped_values_[bucket_index_ * k_bucket_width + slot_index_]};
        }

        const_iterator &operator++() {
            ++slot_index_;
            skip_to_next_occupied();
            return *this;
        }

        const_iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const const_iterator &lhs, const const_iterator &rhs) noexcept {
            return lhs.map_ == rhs.map_ && lhs.bucket_index_ == rhs.bucket_index_ && lhs.slot_index_ == rhs.slot_index_;
        }

        friend bool operator!=(const const_iterator &lhs, const const_iterator &rhs) noexcept { return !(lhs == rhs); }
    };

private:
    hasher_type hasher_{};
    allocator_type alloc_{};
    bucket_allocator_type bucket_alloc_{};
    mapped_allocator_type mapped_alloc_{};

    bucket_type *buckets_ = nullptr;
    mapped_type *mapped_values_ = nullptr;

    size_type bucket_count_ = 0;
    size_type capacity_ = 0; // usable slots (bucket_count_ * k_bucket_width)
    size_type bucket_mask_ = 0;
    size_type size_ = 0;
    size_type growth_limit_ = 0;
    float max_load_factor_ = detail::k_default_max_load_factor;

#ifdef DEBUG
    mutable SizeT total_bucket_fill_ = 0;
    mutable SizeT bucket_fill_ops_ = 0;
    mutable SizeT max_bucket_size_ = 0;
    mutable SizeT rehash_count_ = 0;
    mutable SizeT double_rehash_count_ = 0;
#endif

    [[noreturn]] static void todo_(const char *func) { throw std::logic_error(func); }

    size_type hash_of(const key_type &key) const noexcept { return hasher_(key); }

    size_type index_of(const key_type &key) const noexcept {
        size_type bucket_index = hash_of(key) & bucket_mask_;
        size_type index = bucket_index << 4; // 建立在 bucket width = 16 基础上
        const bucket_type &bucket = buckets_[bucket_index];

        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = bucket.keys[slot_index];
            if (stored_key == key) return index + slot_index;
            if (stored_key == k_unoccupied) return capacity_;
        }
        return capacity_; // 当前 bucket 不再视作不存在（你给的什么垃圾哈希函数（暴论））
    }

    size_type index_of_exist(const key_type &key) const noexcept {
        size_type bucket_index = hash_of(key) & bucket_mask_;
        size_type index = bucket_index << 4;
        const bucket_type &bucket = buckets_[bucket_index];

        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = bucket.keys[slot_index];
            if (stored_key == key) return index + slot_index;
        }
        return capacity_; // 虽然 capacity_ 代表不存在，但是先放着没坏处，防止报错
    }

    void erase_at_index(size_type index) noexcept {
        size_type bucket_index = index / k_bucket_width;
        size_type base_index = bucket_index << 4;
        size_type slot_index = index & k_bucket_width_mask; // 建立在 bucket width = 16 基础上
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        for (; slot_index < k_bucket_width - 1; ++slot_index) {
            size_type next_slot_index = slot_index + 1;
            key_type next_key = keys[next_slot_index];
            if (next_key == k_unoccupied) break;
            keys[slot_index] = keys[next_slot_index];                                                     // 左移 key
            mapped_values_[base_index + slot_index] = std::move(mapped_values_[base_index + slot_index]); // 左移 value
        }
        keys[slot_index] = k_unoccupied;
        if constexpr (!is_trivially_destructible_mapped) std::destroy_at(mapped_values_ + base_index + slot_index);
    }

    void allocate_storage(size_type bucket_count) {
        buckets_ = bucket_alloc_traits::allocate(bucket_alloc_, bucket_count);
        mapped_values_ = mapped_alloc_traits::allocate(mapped_alloc_, capacity_);
        bucket_count_ = bucket_count;
        bucket_mask_ = bucket_count - 1;
        capacity_ = bucket_count << 4;
        growth_limit_ = static_cast<size_type>(max_load_factor_ * static_cast<float>(capacity_));
        // 注意，由于 move old ele 里使用了 memcpy，所以理论上一个 map 只有出生的时候
        // 需要 memset 一次 unoccupied，因此我打算留到构造函数做，别在这里浪费时间
    }

    void deallocate_storage() noexcept {
        if (capacity_ == 0) return;
        bucket_alloc_traits::deallocate(bucket_alloc_, buckets_, bucket_count_);
        mapped_alloc_traits::deallocate(mapped_alloc_, mapped_values_, capacity_);
        buckets_ = nullptr;
        mapped_values_ = nullptr;
        bucket_count_ = 0;
        bucket_mask_ = 0;
        capacity_ = 0;
        growth_limit_ = 0;
    }

    void destroy_all() noexcept {
        if constexpr (is_trivially_destructible_mapped) return;
        for (size_type bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
            auto &keys = buckets_[bucket_index].keys;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                if (keys[slot_index] == k_unoccupied) break;
                size_type index = (bucket_index << 4) + slot_index;
                std::destroy_at(mapped_values_ + index);
            }
        }
    }

    void move_old_elements(Bucket *old_buckets, mapped_type *old_mapped_values, size_type old_bucket_count) {
        // 原本在同一个 bucket 里的 key，到新地方也必然在同一个 bucket
        for (size_type old_bucket_index = 0; old_bucket_index < old_bucket_count; ++old_bucket_index) {
            bucket_type &old_bucket = old_buckets[old_bucket_index];
            auto &keys = old_bucket.keys;
            size_type new_bucket_index = hash_of(keys[0]) & bucket_mask_; // <- 确保 bucket_mask_ 已更新
            std::memcpy(buckets_ + new_bucket_index, old_buckets + old_bucket_index, k_bucket_width * sizeof(key_type));
            size_type old_base_index = old_bucket_index << 4;
            size_type new_base_index = new_bucket_index << 4;
            if constexpr (is_trivially_destructible_mapped) {
                std::memcpy(mapped_values_ + new_base_index, old_mapped_values + old_base_index,
                            k_bucket_width * sizeof(mapped_type));
            } else {
                for (size_type i = 0; i < old_bucket.size; ++i) {
                    std::construct_at(mapped_values_ + new_base_index + i,
                                      std::move(old_mapped_values[old_base_index + i]));
                }
            }
        }
        bucket_alloc_traits::deallocate(bucket_alloc_, old_buckets, old_bucket_count);
        mapped_alloc_traits::deallocate(mapped_alloc_, old_mapped_values, old_bucket_count << 4);
    }

    void double_storage() {
        bucket_type *old_buckets = buckets_;
        mapped_type *old_mapped_values = mapped_values_;
        size_type old_bucket_count = bucket_count_;
        allocate_storage(capacity_ * 2);
        move_old_elements(old_buckets, old_mapped_values, old_bucket_count);
    }

    template <InsertPolicy Policy>
    using insert_return_type =
        std::conditional_t<Policy::return_value,
                           std::conditional_t<Policy::check_dup, std::pair<iterator, bool>, iterator>, void>;

    template <InsertPolicy Policy> auto make_result(size_type index, bool inserted) -> insert_return_type<Policy> {
        if constexpr (Policy::return_value) {
            if constexpr (Policy::check_dup) {
                return {iterator(this, index / k_bucket_width, index & k_bucket_width_mask), inserted};
            } else {
                return iterator(this, index / k_bucket_width, index & k_bucket_width_mask);
            }
        } else {
            return;
        }
    }

public:
    explicit bucket_map_u32(size_type init_size = 1, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), bucket_alloc_(alloc_), mapped_alloc_(alloc_) {
        reserve(init_size);
    }

    bucket_map_u32(std::initializer_list<value_type> init, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), bucket_alloc_(alloc_), mapped_alloc_(alloc_) {
        (void)init;
        // TODO: reserve(init.size()); 然后逐个插入
        todo_("bucket_map_u32::bucket_map_u32(initializer_list)");
    }

    bucket_map_u32(const bucket_map_u32 &other)
        : hasher_(other.hasher_),
          alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)),
          bucket_alloc_(alloc_), mapped_alloc_(alloc_) {
        (void)other;
        // TODO: 深拷贝 buckets_ & mapped_values_
        todo_("bucket_map_u32::bucket_map_u32(const bucket_map_u32&)");
    }

    bucket_map_u32 &operator=(const bucket_map_u32 &other) {
        if (this == &other) return *this;
        (void)other;
        // TODO: 清理自身再拷贝
        todo_("bucket_map_u32::operator=(const bucket_map_u32&)");
    }

    bucket_map_u32(bucket_map_u32 &&other) noexcept
        : hasher_(std::move(other.hasher_)), alloc_(std::move(other.alloc_)),
          bucket_alloc_(std::move(other.bucket_alloc_)), mapped_alloc_(std::move(other.mapped_alloc_)),
          buckets_(other.buckets_), mapped_values_(other.mapped_values_), bucket_count_(other.bucket_count_),
          capacity_(other.capacity_), bucket_mask_(other.bucket_mask_), size_(other.size_),
          growth_limit_(other.growth_limit_), max_load_factor_(other.max_load_factor_)
#ifdef DEBUG
          ,
          total_bucket_fill_(other.total_bucket_fill_), bucket_fill_ops_(other.bucket_fill_ops_),
          max_bucket_size_(other.max_bucket_size_), rehash_count_(other.rehash_count_),
          double_rehash_count_(other.double_rehash_count_)
#endif
    {
        other.buckets_ = nullptr;
        other.mapped_values_ = nullptr;
        other.bucket_count_ = 0;
        other.capacity_ = 0;
        other.bucket_mask_ = 0;
        other.size_ = 0;
        other.growth_limit_ = 0;
#ifdef DEBUG
        other.total_bucket_fill_ = 0;
        other.bucket_fill_ops_ = 0;
        other.max_bucket_size_ = 0;
        other.rehash_count_ = 0;
        other.double_rehash_count_ = 0;
#endif
    }

    bucket_map_u32 &operator=(bucket_map_u32 &&other) noexcept {
        static_assert(bucket_alloc_traits::is_always_equal::value ||
                          bucket_alloc_traits::propagate_on_container_move_assignment::value,
                      "bucket_map_u32 requires allocator that is always_equal or "
                      "propagates on move assignment");
        if (this == &other) return *this;
        (void)other;
        // TODO: 清理自身资源，偷对方指针
        todo_("bucket_map_u32::operator=(bucket_map_u32&&)");
    }

    ~bucket_map_u32() {
        if (bucket_count_ == 0) return;
        // TODO: destroy_all(); deallocate_storage();
        todo_("bucket_map_u32::~bucket_map_u32");
    }

    allocator_type get_allocator() const noexcept { return alloc_; }

    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    size_type bucket_count() const noexcept { return bucket_count_; }
    static constexpr size_type bucket_width() noexcept { return k_bucket_width; }

    float load_factor() const noexcept {
        if (capacity_ == 0) return 0.0f;
        return static_cast<float>(size_) / static_cast<float>(capacity_);
    }

    float max_load_factor() const noexcept { return max_load_factor_; }

    void max_load_factor(float load_factor) {
        if (load_factor < 0.30f) load_factor = 0.30f;
        if (load_factor > 0.95f) load_factor = 0.95f;
        max_load_factor_ = load_factor;
        // TODO: 根据 bucket_count_ 更新 growth_limit_
    }

    void clear() noexcept {
        // TODO: 将所有 bucket 清空，保持容量不变
        todo_("bucket_map_u32::clear");
    }

    void reserve(size_type new_size) {
        if (new_size <= size_) return;
        float need = static_cast<float>(new_size) / max_load_factor_;
        size_type min_capacity = static_cast<size_type>(std::ceil(need));
        if (min_capacity < k_min_capacity) min_capacity = k_min_capacity;
        if (min_capacity <= capacity_) return;
        rehash(min_capacity);
    }

    void shrink_to_fit() {
        // TODO: 按 size_ 重新计算合适 bucket_count_ 并收缩
        todo_("bucket_map_u32::shrink_to_fit");
    }

    void swap(bucket_map_u32 &other) noexcept {
        if (this == &other) return;
        using std::swap;
        // TODO: 交换所有成员
        (void)other;
        todo_("bucket_map_u32::swap");
    }

    void rehash(size_type new_capacity) {
        // TODO: 初始化 bucket_count_ / capacity_ / buckets_ / mapped_values_
        // 预计行为：按元素数量反推 bucket 个数，再向上取 2 的幂
        // TODO: 按元素数量算需要的 bucket 数量并 rehash
        // TODO: new_capacity 表示“元素容量”，内部换算成 bucket_count_
        if (capacity_ == 0) {
            if (new_capacity == 0) return;
            if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity >> 4);
            return;
        }

        if (size_ == 0) {
            deallocate_storage();
            if (new_capacity == 0) return;
            if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity);
            return;
        }

        if (new_capacity <= capacity_) {
            new_capacity = capacity_;
        } else {
            if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
        }

        if (new_capacity == capacity_) return;

        // slot_type *old_slots = slots_;
        mapped_type *old_mapped_values = mapped_values_;
        // size_type old_capacity = capacity_;
        allocate_storage(new_capacity);
        // move_old_elements(old_slots, old_mapped_values, old_capacity);
    }

    iterator begin() noexcept {
        // TODO: 返回指向第一个非空 bucket 第一个元素的迭代器
        todo_("bucket_map_u32::begin");
    }

    const_iterator begin() const noexcept { return cbegin(); }

    const_iterator cbegin() const noexcept {
        // TODO: const 版本 begin
        todo_("bucket_map_u32::cbegin");
    }

    iterator end() noexcept {
        // TODO: end 表示 bucket_index_ == bucket_count_
        todo_("bucket_map_u32::end");
    }

    const_iterator end() const noexcept { return cend(); }

    const_iterator cend() const noexcept {
        // TODO: const 版本 end
        todo_("bucket_map_u32::cend");
    }

    iterator find(const key_type &key) noexcept {
        (void)key;
        // TODO: 使用 index_of
        todo_("bucket_map_u32::find");
    }

    const_iterator find(const key_type &key) const noexcept {
        (void)key;
        // TODO: 使用 index_of
        todo_("bucket_map_u32::find const");
    }

    iterator find_exist(const key_type &key) noexcept {
        (void)key;
        // TODO: index_of_exist
        todo_("bucket_map_u32::find_exist");
    }

    const_iterator find_exist(const key_type &key) const noexcept {
        (void)key;
        todo_("bucket_map_u32::find_exist const");
    }

    bool contains(const key_type &key) const noexcept {
        (void)key;
        // TODO: index_of != capacity_
        todo_("bucket_map_u32::contains");
    }

    mapped_type &at(const key_type &key) {
        (void)key;
        // TODO: 找不到时抛出 out_of_range
        todo_("bucket_map_u32::at");
    }

    const mapped_type &at(const key_type &key) const {
        (void)key;
        todo_("bucket_map_u32::at const");
    }

    mapped_type &operator[](const key_type &key) {
        (void)key;
        // TODO: find or insert default
        todo_("bucket_map_u32::operator[]");
    }

    template <InsertPolicy Policy = default_policy> auto insert(const value_type &kv) -> insert_return_type<Policy> {
        (void)kv;
        // TODO: find / insert
        todo_("bucket_map_u32::insert(const value_type&)");
    }

    template <InsertPolicy Policy = default_policy> auto insert(value_type &&kv) -> insert_return_type<Policy> {
        (void)kv;
        todo_("bucket_map_u32::insert(value_type&&)");
    }

    template <InsertPolicy Policy = default_policy, typename M>
        requires(std::constructible_from<mapped_type, M>)
    auto insert_or_assign(const key_type &key, M &&mapped) -> insert_return_type<Policy> {
        (void)key;
        (void)mapped;
        todo_("bucket_map_u32::insert_or_assign");
    }

    template <InsertPolicy Policy = default_policy, typename M>
        requires(std::constructible_from<mapped_type, M>)
    auto emplace(const key_type &key, M &&mapped) -> insert_return_type<Policy> {
        (void)key;
        (void)mapped;
        todo_("bucket_map_u32::emplace");
    }

    template <InsertPolicy Policy = default_policy, typename... Args>
        requires(std::constructible_from<mapped_type, Args...>)
    auto try_emplace(const key_type &key, Args &&...args) -> insert_return_type<Policy> {
        (void)key;
        (void)std::initializer_list<int>{((void)args, 0)...};
        todo_("bucket_map_u32::try_emplace");
    }

    template <typename M>
        requires(std::constructible_from<mapped_type, M>)
    void overwrite(const key_type &key, M &&mapped) {
        (void)key;
        (void)mapped;
        todo_("bucket_map_u32::overwrite");
    }

    template <std::input_iterator InputIt>
        requires std::convertible_to<std::iter_reference_t<InputIt>, value_type>
    void insert(InputIt first, InputIt last) {
        (void)first;
        (void)last;
        todo_("bucket_map_u32::insert(range)");
    }

    void insert(std::initializer_list<value_type> init) {
        (void)init;
        todo_("bucket_map_u32::insert(initializer_list)");
    }

    size_type erase(const key_type &key) {
        (void)key;
        todo_("bucket_map_u32::erase(key)");
    }

    void erase_exist(const key_type &key) {
        (void)key;
        todo_("bucket_map_u32::erase_exist(key)");
    }

    iterator erase(const_iterator pos) {
        (void)pos;
        todo_("bucket_map_u32::erase(iterator)");
    }

    iterator erase_exist(const_iterator pos) {
        (void)pos;
        todo_("bucket_map_u32::erase_exist(iterator)");
    }

#ifdef DEBUG
    [[nodiscard]] debug_stats get_debug_stats() const noexcept {
        // TODO: 填充真实统计
        debug_stats s{};
        s.size = size_;
        s.capacity = capacity_;
        s.bucket_count = bucket_count_;
        s.rehash_count = rehash_count_;
        s.double_rehash_count = double_rehash_count_;
        s.max_bucket_size = max_bucket_size_;
        s.avg_bucket_fill = bucket_fill_ops_ == 0
                                ? 0.0
                                : static_cast<double>(total_bucket_fill_) / static_cast<double>(bucket_fill_ops_);
        return s;
    }
#endif

    friend bool operator==(const bucket_map_u32 &lhs, const bucket_map_u32 &rhs) {
        (void)lhs;
        (void)rhs;
        // TODO: 按 key 相等 & mapped 相等判断
        todo_("bucket_map_u32::operator==");
    }

    friend void swap(bucket_map_u32 &lhs, bucket_map_u32 &rhs) noexcept(noexcept(lhs.swap(rhs))) { lhs.swap(rhs); }
};
} // namespace mcl