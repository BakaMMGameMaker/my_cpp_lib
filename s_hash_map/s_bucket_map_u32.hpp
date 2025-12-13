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
    static constexpr size_type k_bucket_width = 7; // 7 pairs
    static constexpr size_type k_min_bucket_count = 1;
    static constexpr size_type k_min_capacity = detail::next_power_of_two(k_bucket_width * k_min_bucket_count);
    static constexpr bool move_at_realloc = std::is_nothrow_move_constructible_v<mapped_type> ||
                                            !std::is_copy_constructible_v<mapped_type>; // 搬迁时 move 而不 copy
    static constexpr bool is_trivially_copyable_val = std::is_trivially_copyable_v<mapped_type>;
    static constexpr bool is_trivially_destructible_mapped = std::is_trivially_destructible_v<mapped_type>;

    struct Entry {
        key_type key;
        size_type val_idx;
    };
    static_assert(std::is_trivially_copyable_v<Entry>);
    using entry_type = Entry;

    struct alignas(64) Bucket {
        UInt8 size;
        UInt8 _pad[7]{};
        entry_type entries[k_bucket_width];
    };
    static_assert(sizeof(Bucket) == 64);
    static_assert(std::is_trivially_copyable_v<Bucket>, "Bucket must be trivially copyable");
    using bucket_type = Bucket;

    struct Location {
        size_type bucket_index;
        UInt8 slot_index;
        UInt8 _pad[3]{};
    };
    static_assert(sizeof(Location) == 8);
    using location_type = Location;

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
        const key_type &first;
        mapped_type &second;
        operator value_type() const noexcept { return value_type{first, second}; }
    };

    struct const_value_reference {
        const key_type &first;
        const mapped_type &second;
        operator value_type() const noexcept { return value_type(first, second); }
    };

    struct value_pointer {
        const key_type &first;
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
        mapped_type *vals_ = nullptr;

        size_type bucket_index_ = 0;
        UInt8 slot_index_ = 0;

        iterator(map_type *map, size_type bucket_index, UInt8 slot_index) noexcept
            : map_(map), buckets_(map->buckets_), vals_(map->vals_), bucket_index_(bucket_index),
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
            const entry_type &e = buckets_[bucket_index_].entries[slot_index_];
            return reference{e.key, vals_[e.val_idx]};
        }

        pointer operator->() const {
            const entry_type &e = buckets_[bucket_index_].entries[slot_index_];
            return pointer{e.key, vals_[e.val_idx]};
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
        const mapped_type *vals_ = nullptr;

        size_type bucket_index_ = 0;
        UInt8 slot_index_ = 0;

        const_iterator(const map_type *map, size_type bucket_index, UInt8 slot_index) noexcept
            : map_(map), buckets_(map_->buckets_), vals_(map_->vals_), bucket_index_(bucket_index),
              slot_index_(slot_index) {}

        void skip_to_next_occupied() noexcept {
            // end 时，bucket index = bucket count，slot index = 0
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
            : map_(it.map_), buckets_(it.buckets_), vals_(it.vals_), bucket_index_(it.bucket_index_),
              slot_index_(it.slot_index_) {}

        reference operator*() const {
            const entry_type &e = buckets_[bucket_index_].entries[slot_index_];
            return reference{e.key, vals_[e.val_idx]};
        }

        pointer operator->() const {
            const entry_type &e = buckets_[bucket_index_].entries[slot_index_];
            return pointer{e.key, vals_[e.val_idx]};
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
    mapped_type *vals_ = nullptr;
    location_type *locs_ = nullptr;

    size_type val_cap_ = 0; // 能存储的 mapped value 的最大数量
    size_type bucket_count_ = 0;
    size_type bucket_mask_ = 0;
    size_type size_ = 0;
    size_type growth_limit_ = 0;                                // lf * bucket count 向下取整
    float max_load_factor_ = detail::k_default_max_load_factor; // 每桶平均元素值默认阈值

#ifdef DEBUG
    mutable SizeT total_bucket_fill_ = 0;
    mutable SizeT bucket_fill_ops_ = 0;
    mutable SizeT max_bucket_size_ = 0;
    mutable SizeT rehash_count_ = 0;
    mutable SizeT double_rehash_count_ = 0;
#endif

    size_type hash_of(const key_type &key) const noexcept { return hasher_(key); }

    // 仅分配 mapped + locs
    void alloc_val_storage_for(size_type cap) {
        vals_ = mapped_alloc_traits::allocate(mapped_alloc_, cap);
        locs_ = static_cast<location_type *>(::operator new(static_cast<SizeT>(cap) * sizeof(location_type)));
        val_cap_ = cap;
    }

    // 确保 val_cap_ >= cap，并搬迁 [0..size_) 的 vals + locs
    void ensure_val_cap_for(size_type cap) {
        if (cap <= val_cap_) return;

        // val cap 永远是 2 的次幂
        size_type new_cap = detail::next_power_of_two(cap);
        if (new_cap <= k_min_capacity) new_cap = k_min_capacity;

        mapped_type *new_vals = mapped_alloc_traits::allocate(mapped_alloc_, new_cap);
        location_type *new_locs =
            static_cast<location_type *>(::operator new(static_cast<SizeT>(new_cap) * sizeof(location_type)));

        for (size_type i = 0; i < size_; ++i) {
            if constexpr (move_at_realloc) std::construct_at(new_vals + i, std::move(vals_[i]));
            else std::construct_at(new_vals + i, vals_[i]);
            if constexpr (!is_trivially_destructible_mapped) std::destroy_at(vals_ + i);
        }

        if (size_ != 0) std::memcpy(new_locs, locs_, static_cast<SizeT>(size_) * sizeof(location_type));

        if (val_cap_ != 0) {
            mapped_alloc_traits::deallocate(mapped_alloc_, vals_, val_cap_);
            ::operator delete(locs_);
        }

        vals_ = new_vals;
        locs_ = new_locs;
        val_cap_ = new_cap;
    }

    // 不存在 return bucket count, 0
    std::pair<size_type, UInt8> index_of(const key_type &key, size_type hash) const noexcept {
        size_type bucket_index = hash & bucket_mask_;
        const bucket_type &bucket = buckets_[bucket_index];

        for (UInt8 slot = 0; slot < bucket.size; ++slot) {
            key_type stored_key = bucket.entries[slot].key;
            if (stored_key == key) return {bucket_index, slot};
        }
        return {bucket_count_, 0}; // 不在当前 bucket 视作不存在
    }

    void erase_at(size_type bucket_index, UInt8 slot_index) noexcept {
        bucket_type &bucket = buckets_[bucket_index];
        UInt8 last_slot = bucket.size - 1;
        size_type val_index = bucket.entries[slot_index].val_idx; // 将被删除的 val 的下标

        // 用桶内最后的词条覆盖被删除的词条（逻辑删除）
        if (slot_index != last_slot) {
            bucket.entries[slot_index] = bucket.entries[last_slot];
            size_type last_val_idx = bucket.entries[slot_index].val_idx;
            locs_[last_val_idx] = location_type{bucket_index, slot_index}; // 修正位置信息
        }
        --bucket.size;

        // 用最后一个值覆盖被删除的值
        size_type last_val = size_ - 1;
        if (val_index != last_val) {
            vals_[val_index] = std::move(vals_[last_val]);

            location_type last_loc = locs_[last_val];
            buckets_[last_loc.bucket_index].entries[last_loc.slot_index].val_idx = val_index;
            locs_[val_index] = last_loc;
        }
        if constexpr (!is_trivially_destructible_mapped) std::destroy_at(vals_ + last_val);
        // 我在想可否用一个值来实现 lazy destruction
        // 比如此处 undestructed end = size_
        // 等到需要 construct 的时候，如果目标 index
        // < undestructed end，就直接 move，否则 construct at
        // 搬家的时候，不仅仅摧毁到 size，而是摧毁到 u... end 和 size 的更大者
        --size_;
    }

    // 只分配 bucket 区空间
    void alloc_bucket(size_type bucket_count) {
        bucket_count_ = bucket_count;
        bucket_mask_ = bucket_count - 1;
        growth_limit_ = static_cast<size_type>(max_load_factor_ * static_cast<float>(bucket_count_));

        buckets_ = bucket_alloc_traits::allocate(bucket_alloc_, bucket_count_);
        // Bucket 不构造，所以类内初始值没用，必须 memset
        std::memset(buckets_, 0, static_cast<SizeT>(bucket_count_) * sizeof(bucket_type));
#ifdef DEBUG
        total_bucket_fill_ = 0;
        bucket_fill_ops_ = 0;
        max_bucket_size_ = 0;
#endif
    }

    // 归还全部空间
    void deallocate_storage() noexcept {
        if (bucket_count_ != 0) { bucket_alloc_traits::deallocate(bucket_alloc_, buckets_, bucket_count_); }
        if (val_cap_ != 0) {
            mapped_alloc_traits::deallocate(mapped_alloc_, vals_, bucket_count_);
            ::operator delete(locs_);
        }
        buckets_ = nullptr;
        vals_ = nullptr;
        locs_ = nullptr;
        bucket_count_ = 0;
        bucket_mask_ = 0;
        growth_limit_ = 0;
        val_cap_ = 0;
    }

    // 销毁所有活跃元素
    void destroy_all() noexcept {
        if (size_ == 0) return;
        if constexpr (!is_trivially_destructible_mapped) {
            // undestructed end
            for (size_type i = 0; i < size_; ++i) std::destroy_at(vals_ + i);
        }
        for (size_type i = 0; i < bucket_count_; ++i) buckets_[i].size = 0;
        size_ = 0;
    }

    // 成员更新后调用
    void move_old_elements(Bucket *old_buckets, size_type old_bucket_count) noexcept {
        if (old_bucket_count == 0) return;
        for (size_type old_bidx = 0; old_bidx < old_bucket_count; ++old_bidx) {
            const bucket_type &ob = old_buckets[old_bidx];
            for (size_type slot = 0; slot < ob.size; ++slot) {
                entry_type e = ob.entries[slot];
                size_type new_bidx = hash_of(e.key) & bucket_mask_;
                bucket_type &nb = buckets_[new_bidx];
                // if (nb.size == k_bucket_width)
                //     throw std::runtime_error("bucket is full while moving old elements"); // 理论上不可能
                UInt8 new_slot = nb.size;
                nb.entries[new_slot] = e;
                ++nb.size;
                locs_[e.val_idx] = location_type{new_bidx, new_slot};
#ifdef DEBUG
                if (nb.size > max_bucket_size_) max_bucket_size_ = nb.size;
#endif
            }
        }
        bucket_alloc_traits::deallocate(bucket_alloc_, old_buckets, old_bucket_count);
    }

    // 仅扩 bucket
    void double_storage() {
        bucket_type *old_buckets = buckets_;
        size_type old_bucket_count = bucket_count_;
        alloc_bucket(bucket_count_ == 0 ? 1 : bucket_count_ * 2);
        move_old_elements(old_buckets, old_bucket_count);
#ifdef DEBUG
        ++rehash_count_;
#endif
    }

    template <InsertPolicy Policy>
    using insert_return_type =
        std::conditional_t<Policy::return_value,
                           std::conditional_t<Policy::check_dup, std::pair<iterator, bool>, iterator>, void>;

    template <InsertPolicy Policy>
    auto make_result(size_type bucket_index, UInt8 slot_index, bool inserted) -> insert_return_type<Policy> {
        if constexpr (Policy::return_value) {
            if constexpr (Policy::check_dup) {
                return {iterator(this, bucket_index, slot_index), inserted};
            } else {
                return iterator(this, bucket_index, slot_index);
            }
        } else {
            return;
        }
    }

public:
    bucket_map_u32(size_type init_size = 1, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), bucket_alloc_(alloc_), mapped_alloc_(alloc_) {
        reserve(init_size);
    }

    bucket_map_u32(std::initializer_list<value_type> init, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), bucket_alloc_(alloc_), mapped_alloc_(alloc_) {
        reserve(static_cast<size_type>(init.size()));
        insert<mcl::insert_range>(init.begin(), init.end());
    }

    bucket_map_u32(const bucket_map_u32 &other)
        : hasher_(other.hasher_),
          alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)),
          bucket_alloc_(alloc_), mapped_alloc_(alloc_), buckets_(nullptr), vals_(nullptr), locs_(nullptr), val_cap_(0),
          bucket_count_(0), bucket_mask_(0), size_(0), growth_limit_(0), max_load_factor_(other.max_load_factor_)
#ifdef DEBUG
          ,
          total_bucket_fill_(0), bucket_fill_ops_(0), max_bucket_size_(0), rehash_count_(0), double_rehash_count_(0)
#endif
    {
        if (other.bucket_count_ == 0) return;

        // bucket count 和别人一样
        alloc_bucket(other.bucket_count_);

        if (other.size_ != 0) {
            alloc_val_storage_for(other.size_);
            size_ = other.size_;

            if constexpr (is_trivially_copyable_val) {
                std::memcpy(vals_, other.vals_, static_cast<SizeT>(size_) * sizeof(mapped_type));
            } else {
                for (size_type i = 0; i < size_; ++i) { std::construct_at(vals_ + i, other.vals_[i]); }
            }
            std::memcpy(locs_, other.locs_, static_cast<SizeT>(size_) * sizeof(location_type));
        }
        std::memcpy(buckets_, other.buckets_, static_cast<SizeT>(bucket_count_) * sizeof(bucket_type));
#ifdef DEBUG
        max_bucket_size_ = other.max_bucket_size_;
#endif
    }

    // TODO: 改成和拷贝构造一样的逻辑，避免无意义 hash_of
    bucket_map_u32 &operator=(const bucket_map_u32 &other) {
        if (this == &other) { return *this; }

        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
            if (alloc_ != other.alloc_) {
                destroy_all();
                deallocate_storage();
                alloc_ = other.alloc_;
                bucket_alloc_ = bucket_allocator_type(alloc_);
                mapped_alloc_ = mapped_allocator_type(alloc_);
            }
        }

        hasher_ = other.hasher_;
        max_load_factor_ = other.max_load_factor_;

#ifdef DEBUG
        total_bucket_fill_ = 0;
        bucket_fill_ops_ = 0;
        max_bucket_size_ = 0;
        rehash_count_ = 0;
        double_rehash_count_ = 0;
#endif

        clear();
        reserve(other.size_);
        insert<mcl::fast>(other.begin(), other.end());
        return *this;
    }

    bucket_map_u32(bucket_map_u32 &&other) noexcept
        : hasher_(std::move(other.hasher_)), alloc_(std::move(other.alloc_)),
          bucket_alloc_(std::move(other.bucket_alloc_)), mapped_alloc_(std::move(other.mapped_alloc_)),
          buckets_(other.buckets_), vals_(other.vals_), locs_(other.locs_), val_cap_(other.val_cap_),
          bucket_count_(other.bucket_count_), bucket_mask_(other.bucket_mask_), size_(other.size_),
          growth_limit_(other.growth_limit_), max_load_factor_(other.max_load_factor_)
#ifdef DEBUG
          ,
          total_bucket_fill_(other.total_bucket_fill_), bucket_fill_ops_(other.bucket_fill_ops_),
          max_bucket_size_(other.max_bucket_size_), rehash_count_(other.rehash_count_),
          double_rehash_count_(other.double_rehash_count_)
#endif
    {
        other.buckets_ = nullptr;
        other.vals_ = nullptr;
        other.locs_ = nullptr;
        other.val_cap_ = 0;
        other.bucket_count_ = 0;
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
                      "bucket_map_u32 requires allocator that is always_equal or propagates on move assignment");

        if (this == &other) return *this;

        destroy_all();
        deallocate_storage();

        if constexpr (bucket_alloc_traits::propagate_on_container_move_assignment::value) {
            alloc_ = std::move(other.alloc_);
            bucket_alloc_ = std::move(other.bucket_alloc_);
            mapped_alloc_ = std::move(other.mapped_alloc_);
        }

        hasher_ = std::move(other.hasher_);
        max_load_factor_ = other.max_load_factor_;

        buckets_ = other.buckets_;
        vals_ = other.vals_;
        locs_ = other.locs_;
        val_cap_ = other.val_cap_;
        bucket_count_ = other.bucket_count_;
        bucket_mask_ = other.bucket_mask_;
        size_ = other.size_;
        growth_limit_ = other.growth_limit_;
#ifdef DEBUG
        total_bucket_fill_ = other.total_bucket_fill_;
        bucket_fill_ops_ = other.bucket_fill_ops_;
        max_bucket_size_ = other.max_bucket_size_;
        rehash_count_ = other.rehash_count_;
        double_rehash_count_ = other.double_rehash_count_;
#endif

        other.buckets_ = nullptr;
        other.vals_ = nullptr;
        other.locs_ = nullptr;
        other.val_cap_ = 0;
        other.bucket_count_ = 0;
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
        return *this;
    }

    ~bucket_map_u32() {
        destroy_all();
        deallocate_storage();
    }

    allocator_type get_allocator() const noexcept { return alloc_; }

    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return bucket_count_; }
    size_type growth_limit() const noexcept { return growth_limit_; }
    static constexpr size_type bucket_width() noexcept { return k_bucket_width; }

    float load_factor() const noexcept {
        if (bucket_count_ == 0) return 0.0f;
        return static_cast<float>(size_) / static_cast<float>(bucket_count_);
    }

    float max_load_factor() const noexcept { return max_load_factor_; }

    void max_load_factor(float load_factor) {
        if (load_factor < 0.30f) load_factor = 0.30f;
        if (load_factor > 0.95f) load_factor = 0.95f;
        max_load_factor_ = load_factor;
        growth_limit_ =
            bucket_count_ == 0 ? 0 : static_cast<size_type>(max_load_factor_ * static_cast<float>(bucket_count_));
    }

    void clear() noexcept {
        if (bucket_count_ == 0) size_ = 0;
        if (size_ == 0) return;
        destroy_all();
#ifdef DEBUG
        total_bucket_fill_ = 0;
        bucket_fill_ops_ = 0;
        max_bucket_size_ = 0;
#endif
    }

    void reserve(size_type new_size) {
        if (new_size <= size_) return;

        // values 区只需要能容纳 size 个元素就行，无需预留到 bucket count
        // 我们不需要 values 中留有空位
        ensure_val_cap_for(new_size);

        // bucket 区仍需 max load factor 语义以便即使扩容，减少探测压力
        float need = static_cast<float>(new_size) / max_load_factor_;
        size_type min_bucket_count = static_cast<size_type>(std::ceil(need));
        if (min_bucket_count < k_min_bucket_count) min_bucket_count = k_min_bucket_count;
        if (min_bucket_count <= bucket_count_) return;

        rehash(min_bucket_count);
    }

    void shrink_to_fit() {
        // 这里没动 vals 区也是很诡异的事情，毕竟 bucket 区占地面积只是小头
        if (size_ == 0) deallocate_storage(); // <- bucket count = 0
        if (bucket_count_ == 0) return;

        size_type new_bucket_count = static_cast<size_type>(std::ceil(static_cast<float>(size_) / max_load_factor_));
        if (new_bucket_count <= k_min_capacity) new_bucket_count = k_min_capacity;
        else new_bucket_count = detail::next_power_of_two(new_bucket_count);

        if (new_bucket_count >= bucket_count_) return;

        bucket_type *old_buckets = buckets_;
        size_type old_bucket_count = bucket_count_;

        alloc_bucket(new_bucket_count);
        move_old_elements(old_buckets, old_bucket_count);
    }

    void swap(bucket_map_u32 &other) noexcept {
        if (this == &other) return;
        using std::swap;
        swap(hasher_, other.hasher_);
        swap(alloc_, other.alloc_);
        swap(bucket_alloc_, other.bucket_alloc_);
        swap(mapped_alloc_, other.mapped_alloc_);
        swap(buckets_, other.buckets_);
        swap(vals_, other.vals_);
        swap(locs_, other.locs_);
        swap(val_cap_, other.val_cap_);
        swap(bucket_count_, other.bucket_count_);
        swap(bucket_mask_, other.bucket_mask_);
        swap(size_, other.size_);
        swap(growth_limit_, other.growth_limit_);
        swap(max_load_factor_, other.max_load_factor_);
#ifdef DEBUG
        swap(total_bucket_fill_, other.total_bucket_fill_);
        swap(bucket_fill_ops_, other.bucket_fill_ops_);
        swap(max_bucket_size_, other.max_bucket_size_);
        swap(rehash_count_, other.rehash_count_);
        swap(double_rehash_count_, other.double_rehash_count_);
#endif
    }

    void rehash(size_type new_capacity) {
        // 注意，语义其实相当于期望 bucket count
        size_type new_bucket_count = new_capacity;
        // 只搬 bucket 区
        if (bucket_count_ == 0 || size_ == 0) {
            // 这里是故意的，在 size = 0 的时候 shrink to fit
            // 或者 rehash 0，就回收所有空间，用户必须再次 reserve
            // 或者 rehash 才能插入或者查找，否则就是 UB
            // 这是为了防止在热路径引入 capacity == 0 的分支
            deallocate_storage();
            if (new_bucket_count == 0) return;
            if (new_bucket_count <= k_min_bucket_count) new_bucket_count = k_min_bucket_count;
            else new_bucket_count = detail::next_power_of_two(new_bucket_count);
            alloc_bucket(new_bucket_count);
            return;
        }

        if (new_bucket_count <= bucket_count_) return; // 不允许缩容
        if (new_bucket_count <= k_min_bucket_count) new_bucket_count = k_min_bucket_count;
        else new_bucket_count = detail::next_power_of_two(new_bucket_count);

        bucket_type *old_buckets = buckets_;
        size_type old_bucket_count = bucket_count_;
        alloc_bucket(new_bucket_count);
        move_old_elements(old_buckets, old_bucket_count);
#ifdef DEBUG
        ++rehash_count_;
#endif
    }

    iterator begin() noexcept {
        if (size_ == 0) return end();
        iterator it(this, 0, 0);
        it.skip_to_next_occupied();
        return it;
    }

    const_iterator begin() const noexcept { return cbegin(); }

    const_iterator cbegin() const noexcept {
        if (size_ == 0) return cend();
        const_iterator it(this, 0, 0);
        it.skip_to_next_occupied();
        return it;
    }

    iterator end() noexcept { return iterator(this, bucket_count_, 0); }

    const_iterator end() const noexcept { return cend(); }

    const_iterator cend() const noexcept { return const_iterator(this, bucket_count_, 0); }

    iterator find(const key_type &key) noexcept { return find(key, hash_of(key)); }
    iterator find(const key_type &key, size_type hash) noexcept {
        auto [bucket_index, slot_index] = index_of(key, hash);
        return iterator(this, bucket_index, slot_index);
    }

    const_iterator find(const key_type &key) const noexcept { return find(key, hash_of(key)); }
    const_iterator find(const key_type &key, size_type hash) const noexcept {
        auto [bucket_index, slot_index] = index_of(key, hash);
        return const_iterator(this, bucket_index, slot_index);
    }

    iterator find_exist(const key_type &key) noexcept { return find_exist(key, hash_of(key)); }
    iterator find_exist(const key_type &key, size_type hash) noexcept {
        auto [bucket_index, slot_index] = index_of(key, hash);
        return iterator(this, bucket_index, slot_index);
    }

    const_iterator find_exist(const key_type &key) const noexcept { return find_exist(key, hash_of(key)); }
    const_iterator find_exist(const key_type &key, size_type hash) const noexcept {
        auto [bucket_index, slot_index] = index_of(key, hash);
        return const_iterator(this, bucket_index, slot_index);
    }

    bool contains(const key_type &key) const noexcept { return contains(key, hash_of(key)); }
    bool contains(const key_type &key, size_type hash) const noexcept {
        auto [bucket_index, _] = index_of(key, hash);
        return bucket_index != bucket_count_;
    }

    mapped_type &at(const key_type &key) { return at(key, hash_of(key)); }
    mapped_type &at(const key_type &key, size_type hash) {
        auto [bucket_index, slot_index] = index_of(key, hash);
        if (bucket_index == bucket_count_) throw std::out_of_range("bucket_map_u32::at: key does not exist");
        return vals_[buckets_[bucket_index].entries[slot_index].val_idx];
    }

    const mapped_type &at(const key_type &key) const { return at(key, hash_of(key)); }
    const mapped_type &at(const key_type &key, size_type hash) const {
        auto [bucket_index, slot_index] = index_of(key, hash);
        if (bucket_index == bucket_count_) throw std::out_of_range("bucket_map_u32::at: key does not exist");
        return vals_[buckets_[bucket_index].entries[slot_index].val_idx];
    }

    mapped_type &operator[](const key_type &key) {
        const size_type hash = hash_of(key);

#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::operator[]: bucket_count_ is zero");
#endif
        for (;;) {
            // 查桶
            size_type bidx = hash & bucket_mask_;
            bucket_type &b = buckets_[bidx];

            for (UInt8 slot = 0; slot < b.size; ++slot) {
                entry_type &e = b.entries[slot];
                if (e.key == key) return vals_[e.val_idx];
            }
            // 没找到 key，需要插入 vvv
            ensure_val_cap_for(size_ + 1);

            // 优化点：这里已经确认 key 不存在
            // 但是 continue 又会走一遍查找

            // 过载
            if (size_ >= growth_limit_) {
                double_storage();
                continue; // 原本的 b 已经失效
            }

            // 桶满
            if (b.size == k_bucket_width) {
                rehash(bucket_count_ * 4);
                continue;
            }

            size_type vidx = size_;
            std::construct_at(vals_ + vidx);

            UInt8 new_slot = b.size;
            b.entries[new_slot] = entry_type{key, vidx};
            ++b.size;

            locs_[vidx] = location_type{bidx, new_slot};
            ++size_;
#ifdef DEBUG
            if (bucket.size > max_bucket_size_) max_bucket_size_ = bucket.size;
            total_bucket_fill_ += bucket.size;
            ++bucket_fill_ops_;
#endif
            return vals_[vidx];
        }
    }

    template <InsertPolicy Policy = default_policy> auto insert(const value_type &kv) -> insert_return_type<Policy> {
        return emplace<Policy>(kv.first, hash_of(kv.first), kv.second);
    }

    template <InsertPolicy Policy = default_policy>
    auto insert(const value_type &kv, size_type hash) -> insert_return_type<Policy> {
        return emplace<Policy>(kv.first, hash, kv.second);
    }

    template <InsertPolicy Policy = default_policy> auto insert(value_type &&kv) -> insert_return_type<Policy> {
        return emplace<Policy>(kv.first, hash_of(kv.first), std::move(kv.second));
    }

    template <InsertPolicy Policy = default_policy>
    auto insert(value_type &&kv, size_type hash) -> insert_return_type<Policy> {
        return emplace<Policy>(kv.first, hash, std::move(kv.second));
    }

    template <InsertPolicy Policy = default_policy, typename M>
        requires(std::constructible_from<mapped_type, M>)
    auto insert_or_assign(const key_type &key, M &&mapped) -> insert_return_type<Policy> {
        return insert_or_assign<Policy>(key, hash_of(key), std::forward<M>(mapped));
    }

    template <InsertPolicy Policy = default_policy, typename M>
        requires(std::constructible_from<mapped_type, M>)
    auto insert_or_assign(const key_type &key, size_type hash, M &&mapped) -> insert_return_type<Policy> {
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::insert_or_assign: bucket_count_ is zero");
#endif
        for (;;) {
            size_type bidx = hash & bucket_mask_;
            bucket_type &b = buckets_[bidx];

            if constexpr (Policy::check_dup) {
                for (UInt8 slot = 0; slot < b.size; ++slot) {
                    entry_type &e = b.entries[slot];
                    if (e.key != key) continue;
                    vals_[e.val_idx] = std::forward<M>(mapped);
                    return make_result<Policy>(bidx, slot, false);
                }
            }

            ensure_val_cap_for(size_ + 1);

            if constexpr (Policy::rehash) {
                if (size_ >= growth_limit_) {
                    double_storage();
                    continue;
                }
            }

            if (b.size == k_bucket_width) {
                rehash(bucket_count_ * 4);
                continue;
            }

            size_type vidx = size_;
            std::construct_at(vals_ + vidx, std::forward<M>(mapped));

            UInt8 new_slot = b.size;
            b.entries[new_slot] = entry_type{key, vidx};
            ++b.size;

            locs_[vidx] = location_type{bidx, new_slot};
            ++size_;

#ifdef DEBUG
            if (bucket.size > max_bucket_size_) max_bucket_size_ = bucket.size;
            total_bucket_fill_ += bucket.size;
            ++bucket_fill_ops_;
#endif
            return make_result<Policy>(bidx, new_slot, true);
        }
    }

    template <InsertPolicy Policy = default_policy, typename M>
        requires(std::constructible_from<mapped_type, M>)
    auto emplace(const key_type &key, M &&mapped) -> insert_return_type<Policy> {
        return emplace<Policy>(key, hash_of(key), std::forward<M>(mapped));
    }

    template <InsertPolicy Policy = default_policy, typename M>
        requires(std::constructible_from<mapped_type, M>)
    auto emplace(const key_type &key, size_type hash, M &&mapped) -> insert_return_type<Policy> {
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::emplace: bucket_count_ is zero");
#endif

        for (;;) {
            size_type bidx = hash & bucket_mask_;
            bucket_type &b = buckets_[bidx];

            if constexpr (Policy::check_dup) {
                for (UInt8 slot = 0; slot < b.size; ++slot) {
                    if (b.entries[slot].key == key) { return make_result<Policy>(bidx, slot, false); }
                }
            }

            ensure_val_cap_for(size_ + 1);

            if constexpr (Policy::rehash) {
                if (size_ >= growth_limit_) {
                    double_storage();
                    continue;
                }
            }

            if (b.size == k_bucket_width) {
                rehash(bucket_count_ * 4);
                continue;
            }

            size_type vidx = size_;
            std::construct_at(vals_ + vidx, std::forward<M>(mapped));

            UInt8 new_slot = b.size;
            b.entries[new_slot] = entry_type{key, vidx};
            ++b.size;

            locs_[vidx] = location_type{bidx, new_slot};
            ++size_;

#ifdef DEBUG
            if (bucket.size > max_bucket_size_) max_bucket_size_ = bucket.size;
            total_bucket_fill_ += bucket.size;
            ++bucket_fill_ops_;
#endif
            return make_result<Policy>(bidx, new_slot, true);
        }
    }

    template <InsertPolicy Policy = default_policy, typename... Args>
        requires(std::constructible_from<mapped_type, Args...>)
    auto try_emplace(const key_type &key, Args &&...args) -> insert_return_type<Policy> {
        return try_emplace_with_hash<Policy>(key, hash_of(key), std::forward<Args>(args)...);
    }

    // 嘻嘻，不改名的话可能会有匹配优先级的问题喵，导致 args = {}
    template <InsertPolicy Policy = default_policy, typename... Args>
        requires(std::constructible_from<mapped_type, Args...>)
    auto try_emplace_with_hash(const key_type &key, size_type hash, Args &&...args) -> insert_return_type<Policy> {
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::try_emplace: bucket_count_ is zero");
#endif
        for (;;) {
            size_type bidx = hash & bucket_mask_;
            bucket_type &b = buckets_[bidx];

            if constexpr (Policy::check_dup) {
                for (UInt8 slot = 0; slot < b.size; ++slot) {
                    if (b.entries[slot].key == key) { return make_result<Policy>(bidx, slot, false); }
                }
            }

            ensure_val_cap_for(size_ + 1);

            if constexpr (Policy::rehash) {
                if (size_ >= growth_limit_) {
                    double_storage();
                    continue;
                }
            }

            if (b.size == k_bucket_width) {
                rehash(bucket_count_ * 4);
                continue;
            }

            size_type vidx = size_;
            std::construct_at(vals_ + vidx, std::forward<Args>(args)...);

            UInt8 new_slot = b.size;
            b.entries[new_slot] = entry_type{key, vidx};
            ++b.size;

            locs_[vidx] = location_type{bidx, new_slot};
            ++size_;
#ifdef DEBUG
            if (bucket.size > max_bucket_size_) max_bucket_size_ = bucket.size;
            total_bucket_fill_ += bucket.size;
            ++bucket_fill_ops_;
#endif
            return make_result<Policy>(bidx, new_slot, true);
        }
    }

    // 覆写，返回 key 是否存在
    template <typename M>
        requires(std::constructible_from<mapped_type, M>)
    bool overwrite(const key_type &key, M &&mapped) noexcept {
        return overwrite(key, hash_of(key), std::forward<M>(mapped));
    }

    template <typename M>
        requires(std::constructible_from<mapped_type, M>)
    bool overwrite(const key_type &key, size_type hash, M &&mapped) noexcept {
        size_type bidx = hash & bucket_mask_;
        bucket_type &b = buckets_[bidx];
        for (size_type slot = 0; slot < b.size; ++slot) {
            const entry_type &e = b.entries[slot];
            if (e.key != key) continue;
            size_type vidx = e.val_idx;
            vals_[vidx] = std::forward<M>(mapped);
            return true;
        }
        return false;
    }

    template <InsertPolicy Policy = mcl::insert_range, std::input_iterator InputIt>
        requires std::convertible_to<std::iter_reference_t<InputIt>, value_type>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) { insert<Policy>(*first); }
    }

    template <InsertPolicy Policy = mcl::insert_range> void insert(std::initializer_list<value_type> init) {
        insert<Policy>(init.begin(), init.end());
    }

    size_type erase(const key_type &key) noexcept { return erase(key, hash_of(key)); }

    size_type erase(const key_type &key, size_type hash) noexcept {
        auto [bucket_index, slot_index] = index_of(key, hash);
        if (bucket_index == bucket_count_) return 0;
        erase_at(bucket_index, slot_index);
        return 1;
    }

    // 返回 key 是否存在
    bool erase_exist(const key_type &key) { return erase_exist(key, hash_of(key)); }

    bool erase_exist(const key_type &key, size_type hash) {
        auto [bucket_index, slot_index] = index_of(key, hash);
        if (bucket_index == bucket_count_) return false;
        erase_at(bucket_index, slot_index);
        return true;
    }

    iterator erase(const_iterator pos) {
        if (pos == cend()) return end();
        return erase_exist(pos);
    }

    iterator erase_exist(const_iterator pos) noexcept {
        size_type bucket_index = pos.bucket_index_;
        UInt8 slot_index = pos.slot_index_;
        erase_at(bucket_index, slot_index);
        iterator it(this, bucket_index, slot_index);
        it.skip_to_next_occupied();
        return it;
    }

#ifdef DEBUG
    [[nodiscard]] debug_stats get_debug_stats() const noexcept {
        debug_stats s{};
        s.size = size_;
        s.val_cap_ = val_cap_;
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
        if (&lhs == &rhs) return true;
        if (lhs.size_ != rhs.size_) return false;
        if (lhs.size_ == 0) return true;

        for (size_type bidx = 0; bidx < lhs.bucket_count_; ++bidx) {
            const bucket_type &b = lhs.buckets_[bidx];
            for (UInt8 slot = 0; slot < b.size; ++slot) {
                const entry_type &e = b.entries[slot];
                auto it = rhs.find(e.key);
                if (it == rhs.end()) return false;
                if (!(lhs.vals_[e.val_idx] == it->second)) return false;
            }
        }
        return true;
    }

    friend void swap(bucket_map_u32 &lhs, bucket_map_u32 &rhs) noexcept(noexcept(lhs.swap(rhs))) { lhs.swap(rhs); }
}; // bucket_map_u32
} // namespace mcl