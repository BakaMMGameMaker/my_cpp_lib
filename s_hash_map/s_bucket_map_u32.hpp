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
    static constexpr size_type k_min_capacity = k_bucket_width;
    static constexpr bool is_trivially_copyable_mapped = std::is_trivially_copyable_v<mapped_type>;
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

    size_type hash_of(const key_type &key) const noexcept { return hasher_(key); }

    static constexpr size_type mapped_index(size_type bucket_index, size_type slot_index) noexcept {
        return (bucket_index << 4) + slot_index;
    }

    // 不存在 return bucket count, 0
    std::pair<size_type, size_type> index_of(const key_type &key, size_type hash) const noexcept {
        size_type bucket_index = hash & bucket_mask_;
        const bucket_type &bucket = buckets_[bucket_index];

        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = bucket.keys[slot_index];
            if (stored_key == key) return {bucket_index, slot_index};
            if (stored_key == k_unoccupied) return {bucket_count_, 0};
        }
        return {bucket_count_, 0}; // 不在当前 bucket 视作不存在（嘻嘻）
    }

    std::pair<size_type, size_type> index_of_exist(const key_type &key) const noexcept {
        return index_of_exist(key, hash_of(key));
    }

    std::pair<size_type, size_type> index_of_exist(const key_type &key, size_type hash) const noexcept {
        size_type bucket_index = hash & bucket_mask_;
        const bucket_type &bucket = buckets_[bucket_index];

        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = bucket.keys[slot_index];
            if (stored_key == key) return {bucket_index, slot_index};
        }
        return {bucket_count_, 0};
    }

    void erase_at(size_type bucket_index, size_type slot_index) noexcept {
        size_type base_index = bucket_index << 4;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        size_type write = slot_index;
        for (size_type read = slot_index + 1; read < k_bucket_width; ++read) {
            key_type read_key = keys[read];
            if (read_key == k_unoccupied) break;
            // shift left
            keys[write] = read_key;
            mapped_values_[base_index + write] = std::move(mapped_values_[base_index + read]);
            ++write;
        }
        // 循环退出时，read 最多为 16，write 最多为 15
        keys[write] = k_unoccupied;
        if constexpr (!is_trivially_destructible_mapped) std::destroy_at(mapped_values_ + base_index + write);
        --bucket.size;
        --size_;
    }

    void allocate_storage(size_type bucket_count) {
        bucket_count_ = bucket_count;
        bucket_mask_ = bucket_count - 1;
        capacity_ = bucket_count << 4;
        buckets_ = bucket_alloc_traits::allocate(bucket_alloc_, bucket_count_);
        mapped_values_ = mapped_alloc_traits::allocate(mapped_alloc_, capacity_);
        growth_limit_ = static_cast<size_type>(max_load_factor_ * static_cast<float>(capacity_));

        for (size_type bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
            bucket_type &bucket = buckets_[bucket_index];
            bucket.size = 0;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                bucket.keys[slot_index] = k_unoccupied;
            }
        }
#ifdef DEBUG
        total_bucket_fill_ = 0;
        bucket_fill_ops_ = 0;
        max_bucket_size_ = 0;
#endif
    }

    void deallocate_storage() noexcept {
        if (capacity_ != 0) {
            bucket_alloc_traits::deallocate(bucket_alloc_, buckets_, bucket_count_);
            mapped_alloc_traits::deallocate(mapped_alloc_, mapped_values_, capacity_);
        }
        buckets_ = nullptr;
        mapped_values_ = nullptr;
        bucket_count_ = 0;
        bucket_mask_ = 0;
        capacity_ = 0;
        growth_limit_ = 0;
    }

    // 销毁所有活跃元素，置所有槽位为 unoccupied
    void destroy_all() noexcept {
        for (size_type bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
            bucket_type &bucket = buckets_[bucket_index];
            auto &keys = bucket.keys;
            bucket.size = 0;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                if (keys[slot_index] == k_unoccupied) break;
                keys[slot_index] = k_unoccupied;
                if constexpr (is_trivially_destructible_mapped) continue;
                size_type index = mapped_index(bucket_index, slot_index);
                std::destroy_at(mapped_values_ + index);
            }
        }
        size_ = 0;
    }

    void move_old_elements(Bucket *old_buckets, mapped_type *old_mapped_values, size_type old_bucket_count) {
        if (old_bucket_count == 0) return;
        for (size_type old_bucket_index = 0; old_bucket_index < old_bucket_count; ++old_bucket_index) {
            bucket_type &old_bucket = old_buckets[old_bucket_index];
            auto &keys = old_bucket.keys;
            size_type old_base_index = old_bucket_index << 4;

            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) break;

                size_type old_midx = old_base_index + slot_index;
                mapped_type &old_mapped_value = old_mapped_values[old_midx];

                size_type new_bucket_index = hash_of(stored_key) & bucket_mask_;
                bucket_type &new_bucket = buckets_[new_bucket_index];
                auto &new_keys = new_bucket.keys;
                size_type new_base_index = new_bucket_index << 4;

                if (new_bucket.size == k_bucket_width) throw std::runtime_error("桶满了喵");
                new_keys[new_bucket.size] = stored_key;
                size_type new_midx = new_base_index + new_bucket.size;
                ++new_bucket.size;

                if constexpr (std::is_nothrow_move_constructible_v<mapped_type> ||
                              !std::is_copy_constructible_v<mapped_type>) {
                    std::construct_at(mapped_values_ + new_midx, std::move(old_mapped_value));
                } else {
                    std::construct_at(mapped_values_ + new_midx, old_mapped_value);
                }

                if constexpr (is_trivially_destructible_mapped) continue;

                std::destroy_at(std::addressof(old_mapped_value));
            }
        }
        bucket_alloc_traits::deallocate(bucket_alloc_, old_buckets, old_bucket_count);
        mapped_alloc_traits::deallocate(mapped_alloc_, old_mapped_values, old_bucket_count << 4);
    }

    void double_storage() {
        bucket_type *old_buckets = buckets_;
        mapped_type *old_mapped_values = mapped_values_;
        size_type old_bucket_count = bucket_count_;
        allocate_storage(bucket_count_ == 0 ? 1 : bucket_count_ * 2);
        move_old_elements(old_buckets, old_mapped_values, old_bucket_count);
    }

    template <InsertPolicy Policy>
    using insert_return_type =
        std::conditional_t<Policy::return_value,
                           std::conditional_t<Policy::check_dup, std::pair<iterator, bool>, iterator>, void>;

    template <InsertPolicy Policy>
    auto make_result(size_type bucket_index, size_type slot_index, bool inserted) -> insert_return_type<Policy> {
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
          bucket_alloc_(alloc_), mapped_alloc_(alloc_), buckets_(nullptr), mapped_values_(nullptr), bucket_count_(0),
          capacity_(0), bucket_mask_(0), size_(0), growth_limit_(0), max_load_factor_(other.max_load_factor_)
#ifdef DEBUG
          ,
          total_bucket_fill_(0), bucket_fill_ops_(0), max_bucket_size_(0), rehash_count_(0), double_rehash_count_(0)
#endif
    {
        if (other.bucket_count_ == 0) return;

        allocate_storage(other.bucket_count_);
        size_ = other.size_;

        // 照搬 buckets
        std::memcpy(buckets_, other.buckets_, static_cast<SizeT>(bucket_count_) * sizeof(bucket_type));

        if constexpr (is_trivially_copyable_mapped) {
            std::memcpy(mapped_values_, other.mapped_values_, static_cast<SizeT>(capacity_) * sizeof(mapped_type));
            return;
        }
        // !is trivially copyable vvv
        for (size_type bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
            const bucket_type &src_bucket = other.buckets_[bucket_index];
            const size_type base_index = bucket_index << 4;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type scr_key = src_bucket.keys[slot_index];
                if (scr_key == k_unoccupied) break;
                size_type midx = base_index + slot_index;
                std::construct_at(mapped_values_ + midx, other.mapped_values_[midx]);
            }
        }
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
        if (this == &other) { return *this; }

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
        mapped_values_ = other.mapped_values_;
        bucket_count_ = other.bucket_count_;
        capacity_ = other.capacity_;
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
        return *this;
    }

    ~bucket_map_u32() {
        destroy_all();
        deallocate_storage();
    }

    allocator_type get_allocator() const noexcept { return alloc_; }

    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    size_type bucket_count() const noexcept { return bucket_count_; }
    size_type growth_limit() const noexcept { return growth_limit_; }
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
        growth_limit_ = capacity_ == 0 ? 0 : static_cast<size_type>(max_load_factor_ * static_cast<float>(capacity_));
    }

    void clear() noexcept {
        if (bucket_count_ == 0) size_ = 0;
        if (size_ == 0) return;
        destroy_all();
        size_ = 0;
#ifdef DEBUG
        total_bucket_fill_ = 0;
        bucket_fill_ops_ = 0;
        max_bucket_size_ = 0;
#endif
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
        if (size_ == 0) deallocate_storage(); // <- bucket count = 0
        if (bucket_count_ == 0) return;

        size_type new_capacity = static_cast<size_type>(std::ceil(static_cast<float>(size_) / max_load_factor_));
        if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;
        else new_capacity = detail::next_power_of_two(new_capacity);

        size_type new_bucket_count = new_capacity >> 4;

        if (new_bucket_count >= bucket_count_) return;

        bucket_type *old_buckets = buckets_;
        mapped_type *old_mapped_values = mapped_values_;
        size_type old_bucket_count = bucket_count_;

        allocate_storage(new_bucket_count);
        move_old_elements(old_buckets, old_mapped_values, old_bucket_count);
    }

    void swap(bucket_map_u32 &other) noexcept {
        if (this == &other) return;
        using std::swap;
        swap(hasher_, other.hasher_);
        swap(alloc_, other.alloc_);
        swap(bucket_alloc_, other.bucket_alloc_);
        swap(mapped_alloc_, other.mapped_alloc_);
        swap(buckets_, other.buckets_);
        swap(mapped_values_, other.mapped_values_);
        swap(bucket_count_, other.bucket_count_);
        swap(capacity_, other.capacity_);
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
        if (capacity_ == 0 || size_ == 0) {
            // 这里是故意的，在 size = 0 的时候 shrink to fit
            // 或者 rehash 0，就回收所有空间，用户必须再次 reserve
            // 或者 rehash 才能插入或者查找，否则就是 UB
            // 这是为了防止在热路径引入 capacity == 0 的分支
            deallocate_storage();
            if (new_capacity == 0) return;
            if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity >> 4);
            return;
        }

        if (new_capacity <= capacity_) return; // 不允许缩容
        if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;
        else new_capacity = detail::next_power_of_two(new_capacity);

        bucket_type *old_buckets = buckets_;
        mapped_type *old_mapped_values = mapped_values_;
        size_type old_bucket_count = bucket_count_;
        allocate_storage(new_capacity >> 4);
        move_old_elements(old_buckets, old_mapped_values, old_bucket_count);
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
        auto [bucket_index, slot_index] = index_of_exist(key, hash);
        return iterator(this, bucket_index, slot_index);
    }

    const_iterator find_exist(const key_type &key) const noexcept { return find_exist(key, hash_of(key)); }
    const_iterator find_exist(const key_type &key, size_type hash) const noexcept {
        auto [bucket_index, slot_index] = index_of_exist(key, hash);
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
        return mapped_values_[mapped_index(bucket_index, slot_index)];
    }

    const mapped_type &at(const key_type &key) const { return at(key, hash_of(key)); }
    const mapped_type &at(const key_type &key, size_type hash) const {
        auto [bucket_index, slot_index] = index_of(key, hash);
        if (bucket_index == bucket_count_) throw std::out_of_range("bucket_map_u32::at: key does not exist");
        return mapped_values_[mapped_index(bucket_index, slot_index)];
    }

    mapped_type &operator[](const key_type &key) {
        size_type hash = hash_of(key);
        if (size_ >= growth_limit_) { // 临界元素
            auto [bucket_index, slot_index] = index_of(key, hash);
            if (bucket_index != bucket_count_) return mapped_values_[mapped_index(bucket_index, slot_index)];
            double_storage();
        }
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::operator[]: bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = keys[slot_index];
            if (stored_key == key) {
                size_type midx = mapped_index(bucket_index, slot_index);
                return mapped_values_[midx];
            }
            if (stored_key == k_unoccupied) {
                keys[slot_index] = key;
                size_type midx = mapped_index(bucket_index, slot_index);
                std::construct_at(mapped_values_ + midx);
                ++size_;
                ++bucket.size;
                return mapped_values_[midx];
            }
        }
        // key does not exist but bucket is full
        rehash(capacity_ * 4); // x4 是看心情写的，有空了可以针对这个跑 benchmark
        auto it = emplace<mcl::after_rehash>(key, hash, mapped_type{});
        return it->second;
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
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                auto [bucket_index, slot_index] = index_of(key, hash);
                if (bucket_index != bucket_count_) {
                    size_type midx = mapped_index(bucket_index, slot_index);
                    mapped_values_[midx] = std::forward<M>(mapped);
                    return make_result<Policy>(bucket_index, slot_index, false);
                }
                double_storage();
            }
        }
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::insert_or_assign: bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        if constexpr (Policy::check_dup) {
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) {
                    keys[slot_index] = key;
                    size_type midx = mapped_index(bucket_index, slot_index);
                    std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
                    ++size_;
                    ++bucket.size;
                    return make_result<Policy>(bucket_index, slot_index, true);
                }
                if (stored_key == key) {
                    size_type midx = mapped_index(bucket_index, slot_index);
                    mapped_values_[midx] = std::forward<M>(mapped);
                    return make_result<Policy>(bucket_index, slot_index, false);
                }
            }
            rehash(capacity_ * 4);
            auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
            return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
        } else {
            size_type slot_index = bucket.size;
            if (slot_index == k_bucket_width) {
                rehash(capacity_ * 4);
                auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
                return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
            }
            keys[slot_index] = key;
            size_type midx = mapped_index(bucket_index, slot_index);
            std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
            ++size_;
            ++bucket.size;
            return make_result<Policy>(bucket_index, slot_index, true);
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
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                auto [bucket_index, slot_index] = index_of(key, hash);
                if (bucket_index != bucket_count_) return make_result<Policy>(bucket_index, slot_index, false);
                double_storage();
            }
        }
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::emplace bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        // 查重
        if constexpr (Policy::check_dup) {
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) {
                    keys[slot_index] = key;
                    size_type midx = mapped_index(bucket_index, slot_index);
                    std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
                    ++size_;
                    ++bucket.size;
                    return make_result<Policy>(bucket_index, slot_index, true);
                }
                if (stored_key == key) return make_result<Policy>(bucket_index, slot_index, false);
            }
            rehash(capacity_ * 4);
            auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
            return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
        } else { // 不查重
            size_type slot_index = bucket.size;
            if (slot_index == k_bucket_width) {
                rehash(capacity_ * 4);
                auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
                return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
            }
            keys[slot_index] = key;
            size_type midx = mapped_index(bucket_index, slot_index);
            std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
            ++size_;
            ++bucket.size;
            return make_result<Policy>(bucket_index, slot_index, true);
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
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                auto [bucket_index, slot_index] = index_of(key, hash);
                if (bucket_index != bucket_count_) return make_result<Policy>(bucket_index, slot_index, false);
                double_storage();
            }
        }
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32::try_emplace bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        if constexpr (Policy::check_dup) {
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) {
                    keys[slot_index] = key;
                    size_type midx = mapped_index(bucket_index, slot_index);
                    std::construct_at(mapped_values_ + midx, std::forward<Args>(args)...);
                    ++size_;
                    ++bucket.size;
                    return make_result<Policy>(bucket_index, slot_index, true);
                }
                if (stored_key == key) return make_result<Policy>(bucket_index, slot_index, false);
            }
            rehash(capacity_ * 4);
            auto it = try_emplace_with_hash<mcl::after_rehash>(key, hash, std::forward<Args>(args)...);
            return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
        } else {
            size_type slot_index = bucket.size;
            if (slot_index == k_bucket_width) {
                rehash(capacity_ * 4);
                auto it = try_emplace_with_hash<mcl::after_rehash>(key, hash, std::forward<Args>(args)...);
                return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
            }
            keys[slot_index] = key;
            size_type midx = mapped_index(bucket_index, slot_index);
            std::construct_at(mapped_values_ + midx, std::forward<Args>(args)...);
            ++size_;
            ++bucket.size;
            return make_result<Policy>(bucket_index, slot_index, true);
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
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = keys[slot_index];
            if (stored_key == key) {
                size_type midx = mapped_index(bucket_index, slot_index);
                mapped_values_[midx] = std::forward<M>(mapped);
                return true;
            }
        }
        return false;
    }

    template <InsertPolicy Policy = mcl::insert_range, std::input_iterator InputIt>
        requires std::convertible_to<std::iter_reference_t<InputIt>, value_type>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) insert<Policy>(*first);
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
        auto [bucket_index, slot_index] = index_of_exist(key, hash);
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
        size_type slot_index = pos.slot_index_;
        erase_at(bucket_index, slot_index);
        iterator it(this, bucket_index, slot_index);
        it.skip_to_next_occupied();
        return it;
    }

#ifdef DEBUG
    [[nodiscard]] debug_stats get_debug_stats() const noexcept {
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
        if (&lhs == &rhs) return true;
        if (lhs.size_ != rhs.size_) return false;
        if (lhs.size_ == 0) return true;

        for (size_type bucket_index = 0; bucket_index < lhs.bucket_count_; ++bucket_index) {
            const bucket_type &bucket = lhs.buckets_[bucket_index];
            if (bucket.size == 0) continue;
            const auto &keys = bucket.keys;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) break;
                auto it = rhs.find(stored_key);
                if (it == rhs.end()) return false;
                size_type midx = bucket_index * lhs.k_bucket_width + slot_index;
                const mapped_type &lhs_mapped = lhs.mapped_values_[midx];
                if (!(lhs_mapped == it->second)) return false;
            }
        }
        return true;
    }

    friend void swap(bucket_map_u32 &lhs, bucket_map_u32 &rhs) noexcept(noexcept(lhs.swap(rhs))) { lhs.swap(rhs); }
}; // bucket_map_u32

template <typename MappedType, typename HasherType = detail::FastUInt32Hash,
          typename Alloc = std::allocator<std::pair<UInt32, MappedType>>>
class bucket_map_u32_15keys {
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
    static constexpr size_type k_bucket_width = 15;
    static constexpr size_type k_min_capacity = k_bucket_width;
    static constexpr bool is_trivially_copyable_mapped = std::is_trivially_copyable_v<mapped_type>;
    static constexpr bool is_trivially_destructible_mapped = std::is_trivially_destructible_v<mapped_type>;

    struct Bucket {
        size_type size;
        key_type keys[k_bucket_width];
    };

    static_assert(std::is_trivially_copyable_v<Bucket>, "Bucket must be trivially copyable (size + keys)");
    static_assert(sizeof(Bucket) == 64, "Bucket must be exactly 64 bytes (size + 15 keys)");

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
        friend class bucket_map_u32_15keys;
        friend class const_iterator;

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = bucket_map_u32_15keys::difference_type;
        using reference = bucket_map_u32_15keys::value_reference;
        using pointer = bucket_map_u32_15keys::value_pointer;

    private:
        using map_type = bucket_map_u32_15keys;

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
            ++slot_index_;
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
        friend class bucket_map_u32_15keys;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const bucket_map_u32_15keys::value_type;
        using difference_type = bucket_map_u32_15keys::difference_type;
        using reference = bucket_map_u32_15keys::const_value_reference;
        using pointer = bucket_map_u32_15keys::const_value_pointer;

    private:
        using map_type = const bucket_map_u32_15keys;

        const map_type *map_ = nullptr;
        const bucket_type *buckets_ = nullptr;
        const mapped_type *mapped_values_ = nullptr;

        size_type bucket_index_ = 0;
        size_type slot_index_ = 0;

        const_iterator(const map_type *map, size_type bucket_index, size_type slot_index) noexcept
            : map_(map), buckets_(map_->buckets_), mapped_values_(map_->mapped_values_), bucket_index_(bucket_index),
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

    static constexpr size_type mapped_base_index(size_type bucket_index) noexcept {
        // * 15 == << 4 - * 1
        return (bucket_index << 4) - bucket_index;
    }

    static constexpr size_type mapped_index(size_type bucket_index, size_type slot_index) noexcept {
        return mapped_base_index(bucket_index) + slot_index;
    }

    // 向上取整的除法
    static constexpr size_type ceil_div(size_type a, size_type b) noexcept { return (a + b - 1) / b; }

    static constexpr size_type bucket_count_for_capacity(size_type capacity) noexcept {
        if (capacity == 0) return 0;
        size_type need_buckets = ceil_div(capacity, k_bucket_width);
        return detail::next_power_of_two(need_buckets);
    }

    size_type hash_of(const key_type &key) const noexcept { return hasher_(key); }

    // 不存在 return bucket count, 0
    std::pair<size_type, size_type> index_of(const key_type &key, size_type hash) const noexcept {
        size_type bucket_index = hash & bucket_mask_;
        const bucket_type &bucket = buckets_[bucket_index];

        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = bucket.keys[slot_index];
            if (stored_key == key) return {bucket_index, slot_index};
            if (stored_key == k_unoccupied) return {bucket_count_, 0};
        }
        return {bucket_count_, 0}; // 不在当前 bucket 视作不存在（嘻嘻）
    }

    std::pair<size_type, size_type> index_of_exist(const key_type &key) const noexcept {
        return index_of_exist(key, hash_of(key));
    }

    std::pair<size_type, size_type> index_of_exist(const key_type &key, size_type hash) const noexcept {
        size_type bucket_index = hash & bucket_mask_;
        const bucket_type &bucket = buckets_[bucket_index];

        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = bucket.keys[slot_index];
            if (stored_key == key) return {bucket_index, slot_index};
        }
        return {bucket_count_, 0};
    }

    void erase_at(size_type bucket_index, size_type slot_index) noexcept {
        size_type base_index = mapped_base_index(bucket_index);
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        size_type write = slot_index;
        for (size_type read = slot_index + 1; read < k_bucket_width; ++read) {
            key_type read_key = keys[read];
            if (read_key == k_unoccupied) break;
            // shift left
            keys[write] = read_key;
            mapped_values_[base_index + write] = std::move(mapped_values_[base_index + read]);
            ++write;
        }
        // 循环退出时，read 最多为 16，write 最多为 15
        keys[write] = k_unoccupied;
        if constexpr (!is_trivially_destructible_mapped) std::destroy_at(mapped_values_ + base_index + write);
        --bucket.size;
        --size_;
    }

    void allocate_storage(size_type bucket_count) {
        bucket_count_ = bucket_count;
        bucket_mask_ = bucket_count - 1;
        capacity_ = (bucket_count << 4) - bucket_count; // * 15
        buckets_ = bucket_alloc_traits::allocate(bucket_alloc_, bucket_count_);
        mapped_values_ = mapped_alloc_traits::allocate(mapped_alloc_, capacity_);
        growth_limit_ = static_cast<size_type>(max_load_factor_ * static_cast<float>(capacity_));

        for (size_type bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
            bucket_type &bucket = buckets_[bucket_index];
            bucket.size = 0;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                bucket.keys[slot_index] = k_unoccupied;
            }
        }
#ifdef DEBUG
        total_bucket_fill_ = 0;
        bucket_fill_ops_ = 0;
        max_bucket_size_ = 0;
#endif
    }

    void deallocate_storage() noexcept {
        if (capacity_ != 0) {
            bucket_alloc_traits::deallocate(bucket_alloc_, buckets_, bucket_count_);
            mapped_alloc_traits::deallocate(mapped_alloc_, mapped_values_, capacity_);
        }
        buckets_ = nullptr;
        mapped_values_ = nullptr;
        bucket_count_ = 0;
        bucket_mask_ = 0;
        capacity_ = 0;
        growth_limit_ = 0;
    }

    // 销毁所有活跃元素，置所有槽位为 unoccupied
    void destroy_all() noexcept {
        for (size_type bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
            bucket_type &bucket = buckets_[bucket_index];
            auto &keys = bucket.keys;
            bucket.size = 0;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                if (keys[slot_index] == k_unoccupied) break;
                keys[slot_index] = k_unoccupied;
                if constexpr (is_trivially_destructible_mapped) continue;
                size_type index = mapped_index(bucket_index, slot_index);
                std::destroy_at(mapped_values_ + index);
            }
        }
        size_ = 0;
    }

    void move_old_elements(Bucket *old_buckets, mapped_type *old_mapped_values, size_type old_bucket_count) {
        if (old_bucket_count == 0) return;
        for (size_type old_bucket_index = 0; old_bucket_index < old_bucket_count; ++old_bucket_index) {
            bucket_type &old_bucket = old_buckets[old_bucket_index];
            auto &keys = old_bucket.keys;
            size_type old_base_index = (old_bucket_index << 4) - old_bucket_index;

            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) break;

                size_type old_midx = old_base_index + slot_index;
                mapped_type &old_mapped_value = old_mapped_values[old_midx];

                size_type new_bucket_index = hash_of(stored_key) & bucket_mask_;
                bucket_type &new_bucket = buckets_[new_bucket_index];
                auto &new_keys = new_bucket.keys;
                size_type new_base_index = (new_bucket_index << 4) - new_bucket_index;

                // TODO: 这里的判断可能是不必要的，新桶在理论上一定会有空位留给待插 key
                if (new_bucket.size == k_bucket_width) throw std::runtime_error("桶满了喵");
                new_keys[new_bucket.size] = stored_key;
                size_type new_midx = new_base_index + new_bucket.size;
                ++new_bucket.size;

                if constexpr (std::is_nothrow_move_constructible_v<mapped_type> ||
                              !std::is_copy_constructible_v<mapped_type>) {
                    std::construct_at(mapped_values_ + new_midx, std::move(old_mapped_value));
                } else {
                    std::construct_at(mapped_values_ + new_midx, old_mapped_value);
                }

                if constexpr (is_trivially_destructible_mapped) continue;
                std::destroy_at(std::addressof(old_mapped_value));
            }
        }
        bucket_alloc_traits::deallocate(bucket_alloc_, old_buckets, old_bucket_count);
        mapped_alloc_traits::deallocate(mapped_alloc_, old_mapped_values, (old_bucket_count << 4) - old_bucket_count);
    }

    void double_storage() {
        bucket_type *old_buckets = buckets_;
        mapped_type *old_mapped_values = mapped_values_;
        size_type old_bucket_count = bucket_count_;
        allocate_storage(bucket_count_ == 0 ? 1 : bucket_count_ * 2);
        move_old_elements(old_buckets, old_mapped_values, old_bucket_count);
    }

    template <InsertPolicy Policy>
    using insert_return_type =
        std::conditional_t<Policy::return_value,
                           std::conditional_t<Policy::check_dup, std::pair<iterator, bool>, iterator>, void>;

    template <InsertPolicy Policy>
    auto make_result(size_type bucket_index, size_type slot_index, bool inserted) -> insert_return_type<Policy> {
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
    bucket_map_u32_15keys(size_type init_size = 1, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), bucket_alloc_(alloc_), mapped_alloc_(alloc_) {
        reserve(init_size);
    }

    bucket_map_u32_15keys(std::initializer_list<value_type> init, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), bucket_alloc_(alloc_), mapped_alloc_(alloc_) {
        reserve(static_cast<size_type>(init.size()));
        insert<mcl::insert_range>(init.begin(), init.end());
    }

    bucket_map_u32_15keys(const bucket_map_u32_15keys &other)
        : hasher_(other.hasher_),
          alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)),
          bucket_alloc_(alloc_), mapped_alloc_(alloc_), buckets_(nullptr), mapped_values_(nullptr), bucket_count_(0),
          capacity_(0), bucket_mask_(0), size_(0), growth_limit_(0), max_load_factor_(other.max_load_factor_)
#ifdef DEBUG
          ,
          total_bucket_fill_(0), bucket_fill_ops_(0), max_bucket_size_(0), rehash_count_(0), double_rehash_count_(0)
#endif
    {
        if (other.bucket_count_ == 0) return;

        allocate_storage(other.bucket_count_);
        size_ = other.size_;

        // 照搬 buckets
        std::memcpy(buckets_, other.buckets_, static_cast<SizeT>(bucket_count_) * sizeof(bucket_type));

        if constexpr (is_trivially_copyable_mapped) {
            std::memcpy(mapped_values_, other.mapped_values_, static_cast<SizeT>(capacity_) * sizeof(mapped_type));
            return;
        }
        // !is trivially copyable vvv
        for (size_type bucket_index = 0; bucket_index < bucket_count_; ++bucket_index) {
            const bucket_type &src_bucket = other.buckets_[bucket_index];
            const size_type base_index = bucket_index << 4;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type scr_key = src_bucket.keys[slot_index];
                if (scr_key == k_unoccupied) break;
                size_type midx = base_index + slot_index;
                std::construct_at(mapped_values_ + midx, other.mapped_values_[midx]);
            }
        }
    }

    // TODO: 改成和拷贝构造一样的逻辑，避免无意义 hash_of
    bucket_map_u32_15keys &operator=(const bucket_map_u32_15keys &other) {
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

    bucket_map_u32_15keys(bucket_map_u32_15keys &&other) noexcept
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

    bucket_map_u32_15keys &operator=(bucket_map_u32_15keys &&other) noexcept {
        static_assert(bucket_alloc_traits::is_always_equal::value ||
                          bucket_alloc_traits::propagate_on_container_move_assignment::value,
                      "bucket_map_u32_15keys requires allocator that is always_equal or "
                      "propagates on move assignment");
        if (this == &other) { return *this; }

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
        mapped_values_ = other.mapped_values_;
        bucket_count_ = other.bucket_count_;
        capacity_ = other.capacity_;
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
        return *this;
    }

    ~bucket_map_u32_15keys() {
        destroy_all();
        deallocate_storage();
    }

    allocator_type get_allocator() const noexcept { return alloc_; }

    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    size_type bucket_count() const noexcept { return bucket_count_; }
    size_type growth_limit() const noexcept { return growth_limit_; }
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
        growth_limit_ = capacity_ == 0 ? 0 : static_cast<size_type>(max_load_factor_ * static_cast<float>(capacity_));
    }

    void clear() noexcept {
        if (bucket_count_ == 0) size_ = 0;
        if (size_ == 0) return;
        destroy_all();
        size_ = 0;
#ifdef DEBUG
        total_bucket_fill_ = 0;
        bucket_fill_ops_ = 0;
        max_bucket_size_ = 0;
#endif
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
        if (size_ == 0) deallocate_storage(); // <- bucket count = 0
        if (bucket_count_ == 0) return;
        // size > 0 vvv
        size_type new_capacity = static_cast<size_type>(std::ceil(static_cast<float>(size_) / max_load_factor_));
        if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;

        size_type new_bucket_count = bucket_count_for_capacity(new_capacity);

        if (new_bucket_count >= bucket_count_) return;

        bucket_type *old_buckets = buckets_;
        mapped_type *old_mapped_values = mapped_values_;
        size_type old_bucket_count = bucket_count_;

        allocate_storage(new_bucket_count);
        move_old_elements(old_buckets, old_mapped_values, old_bucket_count);
    }

    void swap(bucket_map_u32_15keys &other) noexcept {
        if (this == &other) return;
        using std::swap;
        swap(hasher_, other.hasher_);
        swap(alloc_, other.alloc_);
        swap(bucket_alloc_, other.bucket_alloc_);
        swap(mapped_alloc_, other.mapped_alloc_);
        swap(buckets_, other.buckets_);
        swap(mapped_values_, other.mapped_values_);
        swap(bucket_count_, other.bucket_count_);
        swap(capacity_, other.capacity_);
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
        if (capacity_ == 0 || size_ == 0) {
            // 这里是故意的，在 size = 0 的时候 shrink to fit
            // 或者 rehash 0，就回收所有空间，用户必须再次 reserve
            // 或者 rehash 才能插入或者查找，否则就是 UB
            // 这是为了防止在热路径引入 capacity == 0 的分支
            deallocate_storage();
            if (new_capacity == 0) return;
            if (new_capacity <= k_min_capacity) new_capacity = k_min_capacity;
            allocate_storage(bucket_count_for_capacity(new_capacity));
            return;
        }

        size_type new_bucket_count = bucket_count_for_capacity(new_capacity);
        if (new_bucket_count <= bucket_count_) return; // 不允许缩容

        bucket_type *old_buckets = buckets_;
        mapped_type *old_mapped_values = mapped_values_;
        size_type old_bucket_count = bucket_count_;
        allocate_storage(new_bucket_count);
        move_old_elements(old_buckets, old_mapped_values, old_bucket_count);
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
        auto [bucket_index, slot_index] = index_of_exist(key, hash);
        return iterator(this, bucket_index, slot_index);
    }

    const_iterator find_exist(const key_type &key) const noexcept { return find_exist(key, hash_of(key)); }
    const_iterator find_exist(const key_type &key, size_type hash) const noexcept {
        auto [bucket_index, slot_index] = index_of_exist(key, hash);
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
        if (bucket_index == bucket_count_) throw std::out_of_range("bucket_map_u32_15keys::at: key does not exist");
        return mapped_values_[mapped_index(bucket_index, slot_index)];
    }

    const mapped_type &at(const key_type &key) const { return at(key, hash_of(key)); }
    const mapped_type &at(const key_type &key, size_type hash) const {
        auto [bucket_index, slot_index] = index_of(key, hash);
        if (bucket_index == bucket_count_) throw std::out_of_range("bucket_map_u32_15keys::at: key does not exist");
        return mapped_values_[mapped_index(bucket_index, slot_index)];
    }

    mapped_type &operator[](const key_type &key) {
        size_type hash = hash_of(key);
        if (size_ >= growth_limit_) { // 临界元素
            auto [bucket_index, slot_index] = index_of(key, hash);
            if (bucket_index != bucket_count_) return mapped_values_[mapped_index(bucket_index, slot_index)];
            double_storage();
        }
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32_15keys::operator[]: bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = keys[slot_index];
            if (stored_key == key) {
                size_type midx = mapped_index(bucket_index, slot_index);
                return mapped_values_[midx];
            }
            if (stored_key == k_unoccupied) {
                keys[slot_index] = key;
                size_type midx = mapped_index(bucket_index, slot_index);
                std::construct_at(mapped_values_ + midx);
                ++size_;
                ++bucket.size;
                return mapped_values_[midx];
            }
        }
        // key does not exist but bucket is full
        rehash(capacity_ * 4); // x4 是看心情写的，有空了可以针对这个跑 benchmark
        auto it = emplace<mcl::after_rehash>(key, hash, mapped_type{});
        return it->second;
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
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                auto [bucket_index, slot_index] = index_of(key, hash);
                if (bucket_index != bucket_count_) {
                    size_type midx = mapped_index(bucket_index, slot_index);
                    mapped_values_[midx] = std::forward<M>(mapped);
                    return make_result<Policy>(bucket_index, slot_index, false);
                }
                double_storage();
            }
        }
#ifdef DEBUG
        if (bucket_count_ == 0)
            throw std::runtime_error("bucket_map_u32_15keys::insert_or_assign: bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        if constexpr (Policy::check_dup) {
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) {
                    keys[slot_index] = key;
                    size_type midx = mapped_index(bucket_index, slot_index);
                    std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
                    ++size_;
                    ++bucket.size;
                    return make_result<Policy>(bucket_index, slot_index, true);
                }
                if (stored_key == key) {
                    size_type midx = mapped_index(bucket_index, slot_index);
                    mapped_values_[midx] = std::forward<M>(mapped);
                    return make_result<Policy>(bucket_index, slot_index, false);
                }
            }
            rehash(capacity_ * 4);
            auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
            return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
        } else {
            size_type slot_index = bucket.size;
            if (slot_index == k_bucket_width) {
                rehash(capacity_ * 4);
                auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
                return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
            }
            keys[slot_index] = key;
            size_type midx = mapped_index(bucket_index, slot_index);
            std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
            ++size_;
            ++bucket.size;
            return make_result<Policy>(bucket_index, slot_index, true);
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
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                auto [bucket_index, slot_index] = index_of(key, hash);
                if (bucket_index != bucket_count_) return make_result<Policy>(bucket_index, slot_index, false);
                double_storage();
            }
        }
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32_15keys::emplace bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        // 查重
        if constexpr (Policy::check_dup) {
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) {
                    keys[slot_index] = key;
                    size_type midx = mapped_index(bucket_index, slot_index);
                    std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
                    ++size_;
                    ++bucket.size;
                    return make_result<Policy>(bucket_index, slot_index, true);
                }
                if (stored_key == key) return make_result<Policy>(bucket_index, slot_index, false);
            }
            rehash(capacity_ * 4);
            auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
            return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
        } else { // 不查重
            size_type slot_index = bucket.size;
            if (slot_index == k_bucket_width) {
                rehash(capacity_ * 4);
                auto it = emplace<mcl::after_rehash>(key, hash, std::forward<M>(mapped));
                return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
            }
            keys[slot_index] = key;
            size_type midx = mapped_index(bucket_index, slot_index);
            std::construct_at(mapped_values_ + midx, std::forward<M>(mapped));
            ++size_;
            ++bucket.size;
            return make_result<Policy>(bucket_index, slot_index, true);
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
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                auto [bucket_index, slot_index] = index_of(key, hash);
                if (bucket_index != bucket_count_) return make_result<Policy>(bucket_index, slot_index, false);
                double_storage();
            }
        }
#ifdef DEBUG
        if (bucket_count_ == 0) throw std::runtime_error("bucket_map_u32_15keys::try_emplace bucket_count_ is zero");
#endif
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        if constexpr (Policy::check_dup) {
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) {
                    keys[slot_index] = key;
                    size_type midx = mapped_index(bucket_index, slot_index);
                    std::construct_at(mapped_values_ + midx, std::forward<Args>(args)...);
                    ++size_;
                    ++bucket.size;
                    return make_result<Policy>(bucket_index, slot_index, true);
                }
                if (stored_key == key) return make_result<Policy>(bucket_index, slot_index, false);
            }
            rehash(capacity_ * 4);
            auto it = try_emplace_with_hash<mcl::after_rehash>(key, hash, std::forward<Args>(args)...);
            return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
        } else {
            size_type slot_index = bucket.size;
            if (slot_index == k_bucket_width) {
                rehash(capacity_ * 4);
                auto it = try_emplace_with_hash<mcl::after_rehash>(key, hash, std::forward<Args>(args)...);
                return make_result<Policy>(it.bucket_index_, it.slot_index_, true);
            }
            keys[slot_index] = key;
            size_type midx = mapped_index(bucket_index, slot_index);
            std::construct_at(mapped_values_ + midx, std::forward<Args>(args)...);
            ++size_;
            ++bucket.size;
            return make_result<Policy>(bucket_index, slot_index, true);
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
        size_type bucket_index = hash & bucket_mask_;
        bucket_type &bucket = buckets_[bucket_index];
        auto &keys = bucket.keys;
        for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
            key_type stored_key = keys[slot_index];
            if (stored_key == key) {
                size_type midx = mapped_index(bucket_index, slot_index);
                mapped_values_[midx] = std::forward<M>(mapped);
                return true;
            }
        }
        return false;
    }

    template <InsertPolicy Policy = mcl::insert_range, std::input_iterator InputIt>
        requires std::convertible_to<std::iter_reference_t<InputIt>, value_type>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) insert<Policy>(*first);
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
        auto [bucket_index, slot_index] = index_of_exist(key, hash);
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
        size_type slot_index = pos.slot_index_;
        erase_at(bucket_index, slot_index);
        iterator it(this, bucket_index, slot_index);
        it.skip_to_next_occupied();
        return it;
    }

#ifdef DEBUG
    [[nodiscard]] debug_stats get_debug_stats() const noexcept {
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

    friend bool operator==(const bucket_map_u32_15keys &lhs, const bucket_map_u32_15keys &rhs) {
        if (&lhs == &rhs) return true;
        if (lhs.size_ != rhs.size_) return false;
        if (lhs.size_ == 0) return true;

        for (size_type bucket_index = 0; bucket_index < lhs.bucket_count_; ++bucket_index) {
            const bucket_type &bucket = lhs.buckets_[bucket_index];
            if (bucket.size == 0) continue;
            const auto &keys = bucket.keys;
            for (size_type slot_index = 0; slot_index < k_bucket_width; ++slot_index) {
                key_type stored_key = keys[slot_index];
                if (stored_key == k_unoccupied) break;
                auto it = rhs.find(stored_key);
                if (it == rhs.end()) return false;
                size_type midx = bucket_index * lhs.k_bucket_width + slot_index;
                const mapped_type &lhs_mapped = lhs.mapped_values_[midx];
                if (!(lhs_mapped == it->second)) return false;
            }
        }
        return true;
    }

    friend void swap(bucket_map_u32_15keys &lhs, bucket_map_u32_15keys &rhs) noexcept(noexcept(lhs.swap(rhs))) {
        lhs.swap(rhs);
    }
}; // bucket_map_u32_15keys
} // namespace mcl