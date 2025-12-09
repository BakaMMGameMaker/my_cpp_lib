#pragma once
#include "s_alias.h"
#include "s_detail.h"
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace mcl {

template <typename P>
concept InsertPolicy = requires {
    { P::return_value } -> std::convertible_to<bool>; // 是否返回值
    { P::rehash } -> std::convertible_to<bool>;       // 是否触发扩容
    { P::check_dup } -> std::convertible_to<bool>;    // 是否检查重复键
};

struct default_policy {
    static constexpr bool return_value = true;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = true;
};

// 不返回值
struct no_return {
    static constexpr bool return_value = false;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = true;
};

// 不扩容
struct no_rehash {
    static constexpr bool return_value = true;
    static constexpr bool rehash = false;
    static constexpr bool check_dup = true;
};

// 不查重
struct no_check_dup {
    static constexpr bool return_value = true;
    static constexpr bool rehash = true;
    static constexpr bool check_dup = false;
};

// 仅插入元素
struct fast {
    static constexpr bool return_value = false;
    static constexpr bool rehash = false;
    static constexpr bool check_dup = false;
};

template <typename MappedType, typename HasherType = detail::FastUInt32Hash,
          typename Alloc = std::allocator<std::pair<UInt32, MappedType>>> // TODO：Alloc 的默认参数可能需要更换
class flat_hash_map_u32 {
public:
    using key_type = UInt32;
    using mapped_type = MappedType;
    using value_type = std::pair<key_type, mapped_type>;
    using size_type = SizeT;
    using difference_type = std::ptrdiff_t;
    using hasher_type = HasherType;
    using key_equal_type = std::equal_to<key_type>;
    using allocator_type = Alloc;

private:
    struct Slot {
        key_type key = k_unoccupied;
    };

    using slot_type = Slot;
    using slot_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<slot_type>;
    using slot_alloc_traits = std::allocator_traits<slot_allocator_type>;
    using value_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<value_type>;
    using value_alloc_traits = std::allocator_traits<value_allocator_type>;

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

public:
    class const_iterator;
    class iterator {
        friend class flat_hash_map_u32;
        friend class const_iterator;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = flat_hash_map_u32::value_type;
        using difference_type = flat_hash_map_u32::difference_type;

    private:
        using map_type = flat_hash_map_u32;
        using slot_type = map_type::slot_type;

        map_type *map_ = nullptr;
        size_type index_ = 0;
        size_type capacity_ = 0;
        slot_type *slots_ = nullptr;
        value_type *values_ = nullptr;

        iterator(map_type *map, size_type index) noexcept
            : map_(map), index_(index), capacity_(map->capacity_), slots_(map->slots_), values_(map->values_) {}

        void skip_to_next_occupied() noexcept {
            while (index_ < capacity_) {
                if (slots_[index_].key != k_unoccupied) break;
                ++index_;
            }
        }

    public:
        iterator() noexcept = default;

        value_type &operator*() const noexcept { return values_[index_]; }

        value_type *operator->() const noexcept { return std::addressof(values_[index_]); }

        iterator &operator++() noexcept {
            ++index_;
            skip_to_next_occupied();
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
    };

    class const_iterator {
        friend class flat_hash_map_u32;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const flat_hash_map_u32::value_type;
        using difference_type = flat_hash_map_u32::difference_type;

    private:
        using map_type = const flat_hash_map_u32;
        using slot_type = const map_type::slot_type;

        const map_type *map_ = nullptr;
        size_type index_ = 0;
        size_type capacity_ = 0;
        const slot_type *slots_ = nullptr;
        const value_type *values_ = nullptr;

        const_iterator(const map_type *map, size_type index) noexcept
            : map_(map), index_(index), capacity_(map->capacity_), slots_(map->slots_), values_(map->values_) {}

        void skip_to_next_occupied() noexcept {
            while (index_ < capacity_) {
                if (slots_[index_].key != k_unoccupied) break;
                ++index_;
            }
        }

    public:
        const_iterator() noexcept = default;

        const_iterator(const iterator &it) noexcept
            : map_(it.map_), index_(it.index_), capacity_(it.capacity_), slots_(it.slots_), values_(it.values_) {}

        const value_type &operator*() const noexcept { return values_[index_]; }

        const value_type *operator->() const noexcept { return std::addressof(values_[index_]); }

        const_iterator &operator++() noexcept {
            ++index_;
            skip_to_next_occupied();
            return *this;
        }

        const_iterator operator++(int) noexcept {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const const_iterator &lhs, const const_iterator &rhs) noexcept {
            return lhs.map_ == rhs.map_ && lhs.index_ == rhs.index_;
        }

        friend bool operator!=(const const_iterator &lhs, const const_iterator &rhs) noexcept { return !(lhs == rhs); }
    };

private:
    // return type alias
    // emplace 返回类型
    template <InsertPolicy Policy>
    using insert_return_type =
        std::conditional_t<Policy::return_value,
                           std::conditional_t<Policy::check_dup, std::pair<iterator, bool>, iterator>, void>;

    // find or insert kv 返回类型
    template <InsertPolicy Policy>
    using find_or_insert_kv_return_type =
        std::conditional_t<Policy::return_value,
                           std::conditional_t<Policy::check_dup, std::pair<size_type, bool>, size_type>, void>;

public:
    explicit flat_hash_map_u32(size_type init_size = 1, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), slot_alloc_(alloc_), value_alloc_(alloc_) {
        reserve(init_size);
    }

    flat_hash_map_u32(std::initializer_list<value_type> init, const Alloc &alloc = Alloc{})
        : hasher_(), alloc_(alloc), slot_alloc_(alloc_), value_alloc_(alloc_) {
        reserve(init.size());
        for (const auto &kv : init) { insert(kv); }
    }

    flat_hash_map_u32(const flat_hash_map_u32 &other)
        : hasher_(other.hasher_),
          alloc_(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.alloc_)),
          slot_alloc_(alloc_), value_alloc_(alloc_) {
        if (other.size_ == 0) return;
        rehash(other.capacity_);
        for (size_type index = 0; index < other.capacity_; ++index) {
            if (other.slots_[index].key == k_unoccupied) continue;
            insert(other.values_[index]);
        }
    }

    flat_hash_map_u32 &operator=(const flat_hash_map_u32 &other) {
        if (this == &other) return *this;
        clear();
        if constexpr (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
            alloc_ = other.alloc_;
            slot_alloc_ = other.slot_alloc_;
            value_alloc_ = other.value_alloc_;
        }
        hasher_ = other.hasher_;
        if (other.size_ == 0) return *this;
        rehash(other.capacity_);
        for (size_type index = 0; index < other.capacity_; ++index) {
            if (other.slots_[index].key == k_unoccupied) continue;
            insert(other.values_[index]);
        }
        return *this;
    }

    // 照搬布局
    flat_hash_map_u32(flat_hash_map_u32 &&other) noexcept
        : hasher_(std::move(other.hasher_)), alloc_(std::move(other.alloc_)), slot_alloc_(std::move(other.slot_alloc_)),
          value_alloc_(std::move(other.value_alloc_)), slots_(other.slots_), values_(other.values_),
          capacity_(other.capacity_), bucket_mask_(other.bucket_mask_), size_(other.size_),
          growth_limit_(other.growth_limit_), max_load_factor_(other.max_load_factor_)
#ifdef DEBUG
          ,
          total_probes_(other.total_probes_), probe_ops_(other.probe_ops_), max_probe_len_(other.max_probe_len_),
          rehash_count_(other.rehash_count_), double_rehash_count_(other.double_rehash_count_)
#endif
    {
        other.slots_ = nullptr;
        other.values_ = nullptr;
        other.capacity_ = 0;
        other.bucket_mask_ = 0;
        other.size_ = 0;
        other.growth_limit_ = 0;
#ifdef DEBUG
        other.total_probes_ = 0;
        other.probe_ops_ = 0;
        other.max_probe_len_ = 0;
        other.rehash_count_ = 0;
        other.double_rehash_count_ = 0;
#endif
    }

    flat_hash_map_u32 &operator=(flat_hash_map_u32 &&other) noexcept {
        static_assert(slot_alloc_traits::is_always_equal::value ||
                          slot_alloc_traits::propagate_on_container_move_assignment::value,
                      "flat_hash_map_u32 requires allocator that is always_equal or propagates on move assignment");
        if (this == &other) return *this;
        if (capacity_ > 0) {
            destroy_all();
            deallocate_storage();
        }
        if constexpr (slot_alloc_traits::propagate_on_container_move_assignment::value) {
            alloc_ = std::move(other.alloc_);
            slot_alloc_ = std::move(other.slot_alloc_);
            value_alloc_ = std::move(other.value_alloc_);
        }
        hasher_ = std::move(other.hasher_);
        slots_ = other.slots_;
        values_ = other.values_;
        capacity_ = other.capacity_;
        bucket_mask_ = other.bucket_mask_;
        size_ = other.size_;
        growth_limit_ = other.growth_limit_;
        max_load_factor_ = other.max_load_factor_;
#ifdef DEBUG
        total_probes_ = other.total_probes_;
        probe_ops_ = other.probe_ops_;
        max_probe_len_ = other.max_probe_len_;
        rehash_count_ = other.rehash_count_;
        double_rehash_count_ = other.double_rehash_count_;
#endif

        other.slots_ = nullptr;
        other.values_ = nullptr;
        other.capacity_ = 0;
        other.bucket_mask_ = 0;
        other.size_ = 0;
        other.growth_limit_ = 0;
#ifdef DEBUG
        other.total_probes_ = 0;
        other.probe_ops_ = 0;
        other.max_probe_len_ = 0;
        other.rehash_count_ = 0;
        other.double_rehash_count_ = 0;
#endif
        return *this;
    }

    ~flat_hash_map_u32() {
        if (capacity_ == 0) return;
        destroy_all();
        deallocate_storage();
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

    void max_load_factor(float load_factor) {
        if (load_factor < 0.50f) load_factor = 0.50f;
        if (load_factor > 0.95f) load_factor = 0.95f;
        max_load_factor_ = load_factor;
        growth_limit_ = static_cast<size_type>(max_load_factor_ * static_cast<float>(capacity_));
    }

    void clear() noexcept {
        for (size_type index = 0; index < capacity_; ++index) {
            if (slots_[index].key != k_unoccupied) {
                slots_[index].key = k_unoccupied;
                std::destroy_at(values_ + index);
            }
        }
        size_ = 0;
    }

    void reserve(size_type new_size) {
        float need = static_cast<float>(new_size) / max_load_factor_;
        size_type min_capacity = static_cast<size_type>(std::ceil(need));
        if (min_capacity < detail::k_min_capacity) min_capacity = detail::k_min_capacity;
        if (min_capacity <= capacity_) return;
        rehash(min_capacity);
    }

    void shrink_to_fit() {
        if (size_ == 0) {
            if (capacity_ == 0) return;
            deallocate_storage();
            slots_ = nullptr;
            values_ = nullptr;
            capacity_ = 0;
            bucket_mask_ = 0;
            growth_limit_ = 0;
            return;
        }
        size_type new_capacity = static_cast<size_type>(std::ceil(static_cast<float>(size_) / max_load_factor_));
        if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
        else new_capacity = detail::next_power_of_two(new_capacity);

        if (new_capacity == capacity_) return;

        slot_type *old_slots = slots_;
        value_type *old_values = values_;
        size_type old_capacity = capacity_;
        allocate_storage(new_capacity);
        move_old_elements(old_slots, old_values, old_capacity);
    }

    void swap(flat_hash_map_u32 &other) noexcept {
        static_assert(std::allocator_traits<allocator_type>::propagate_on_container_swap::value ||
                          std::allocator_traits<allocator_type>::is_always_equal::value,
                      "flat_hash_map_u32 requires allocator that is always_equal or propagates on swap");
        if (this == &other) return;
        using std::swap;
        if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value) {
            swap(alloc_, other.alloc_);
            swap(slot_alloc_, other.slot_alloc_);
            swap(value_alloc_, other.value_alloc_);
        }
        swap(hasher_, other.hasher_);
        swap(slots_, other.slots_);
        swap(values_, other.values_);
        swap(capacity_, other.capacity_);
        swap(bucket_mask_, other.bucket_mask_);
        swap(size_, other.size_);
        swap(growth_limit_, other.growth_limit_);
        swap(max_load_factor_, other.max_load_factor_);
#ifdef DEBUG
        swap(total_probes_, other.total_probes_);
        swap(probe_ops_, other.probe_ops_);
        swap(max_probe_len_, other.max_probe_len_);
        swap(rehash_count_, other.rehash_count_);
        swap(double_rehash_count_, other.double_rehash_count_);
#endif
    }

    void rehash(size_type new_capacity) {
        if (capacity_ == 0) {
            if (new_capacity == 0) return;
            if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity);
#ifdef DEBUG
            ++rehash_count_;
#endif
            return;
        }

        if (size_ == 0) {
            destroy_all();
            deallocate_storage();
            if (new_capacity == 0) {
                slots_ = nullptr;
                values_ = nullptr;
                capacity_ = 0;
                bucket_mask_ = 0;
                growth_limit_ = 0;
#ifdef DEBUG
                ++rehash_count_;
#endif
                return;
            }
            if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
            allocate_storage(new_capacity);
#ifdef DEBUG
            ++rehash_count_;
#endif
            return;
        }

        if (new_capacity <= capacity_) {
            new_capacity = capacity_;
        } else {
            if (new_capacity <= detail::k_min_capacity) new_capacity = detail::k_min_capacity;
            else new_capacity = detail::next_power_of_two(new_capacity);
        }

        if (new_capacity == capacity_) return;

        slot_type *old_slots = slots_;
        value_type *old_values = values_;
        size_type old_capacity = capacity_;
        allocate_storage(new_capacity);
#ifdef DEBUG
        ++rehash_count_;
#endif
        move_old_elements(old_slots, old_values, old_capacity);
    }

    iterator begin() noexcept {
        if (!slots_ || size_ == 0) return iterator(this, capacity_);
        iterator it(this, 0);
        it.skip_to_next_occupied();
        return it;
    }

    const_iterator begin() const noexcept { return cbegin(); }
    const_iterator cbegin() const noexcept {
        if (!slots_ || size_ == 0) return const_iterator(this, capacity_);
        const_iterator it(this, 0);
        it.skip_to_next_occupied();
        return it;
    }

    iterator end() noexcept { return iterator(this, capacity_); }
    const_iterator end() const noexcept { return cend(); }
    const_iterator cend() const noexcept { return const_iterator(this, capacity_); }

    // 不检查是否在查找哨兵值
    iterator find(const key_type &key) noexcept { return iterator(this, index_of(key)); }

    const_iterator find(const key_type &key) const noexcept { return const_iterator(this, index_of(key)); }

    bool contains(const key_type &key) const noexcept { return index_of(key) != capacity_; }

    mapped_type &at(const key_type &key) {
        auto it = find(key);
        if (it == end()) throw std::out_of_range("flat_hash_map::at: key not found");
        return it->second;
    }

    const mapped_type &at(const key_type &key) const {
        auto it = find(key);
        if (it == cend()) throw std::out_of_range("flat_hash_map::at: key not found");
        return it->second;
    }

    mapped_type &operator[](const key_type &key) {
        // 不检查是否插入哨兵值
        size_type index = find_or_insert_default(key);
        return values_[index].second;
    }

    std::pair<iterator, bool> insert(const value_type &kv) { return emplace(kv.first, kv.second); }

    std::pair<iterator, bool> insert(value_type &&kv) { return emplace(kv.first, std::move(kv.second)); }

    template <typename M>
        requires(std::constructible_from<mapped_type, M>)
    std::pair<iterator, bool> insert_or_assign(const key_type &key, M &&mapped) {
        auto [index, inserted] = find_and_insert_or_assign(key, std::forward<M>(mapped));
        return {iterator(this, index), inserted};
    }

    template <InsertPolicy Policy = default_policy, typename M>
        requires(std::constructible_from<mapped_type, M>)
    auto emplace(const key_type &key, M &&mapped) -> insert_return_type<Policy> {
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                size_type existing = index_of(key);
                if (existing != capacity_) return make_result<Policy>(existing, false);
                double_storage();
            }
        }
        size_type index = static_cast<size_type>(hash_of(key)) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            slot_type &slot = slots_[index];
            key_type stored_key = slot.key;
            if (stored_key == k_unoccupied) {
                slot.key = key;
                std::construct_at(values_ + index, std::piecewise_construct, std::forward_as_tuple(key),
                                  std::forward_as_tuple(std::forward<M>(mapped)));
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return make_result<Policy>(index, true);
            }
            if constexpr (Policy::check_dup) {
                if (stored_key == key) {
#ifdef DEBUG
                    record_probe(probe_len);
#endif
                    return make_result<Policy>(index, false);
                }
            }
            index = (index + 1) & bucket_mask_;
        }
    }

    template <InsertPolicy Policy = default_policy, typename... Args>
        requires(std::constructible_from<mapped_type, Args...>)
    auto try_emplace(const key_type &key, Args &&...args) -> insert_return_type<Policy> {
        if constexpr (Policy::rehash) {
            if (size_ >= growth_limit_) {
                size_type existing = index_of(key);
                if (existing != capacity_) return make_result<Policy>(existing, false);
                double_storage();
            }
        }
        size_type index = static_cast<size_type>(hash_of(key)) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            slot_type &slot = slots_[index];
            key_type stored_key = slot.key;
            if (stored_key == k_unoccupied) {
                slot.key = key;
                std::construct_at(values_ + index, std::piecewise_construct, std::forward_as_tuple(key),
                                  std::forward_as_tuple(std::forward<Args>(args)...));
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return make_result<Policy>(index, true);
            }
            if constexpr (Policy::check_dup) {
                if (stored_key == key) {
#ifdef DEBUG
                    record_probe(probe_len);
#endif
                    return make_result<Policy>(index, false);
                }
            }
            index = (index + 1) & bucket_mask_;
        }
    }

    template <std::input_iterator InputIt>
        requires std::convertible_to<std::iter_reference_t<InputIt>, value_type>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) insert(*first);
    }

    void insert(std::initializer_list<value_type> init) { insert(init.begin(), init.end()); }

    size_type erase(const key_type &key) {
        size_type index = index_of(key);
        if (index == capacity_) return 0;
        erase_at_index(index);
        return 1;
    }

    // 不检查 pos 是否指向合法槽位
    iterator erase(const_iterator pos) {
        if (pos == end()) return end();
        size_type index = pos.index_;
        erase_at_index(index);
        iterator it(this, index);
        it.skip_to_next_occupied();
        return it;
    }

#ifdef DEBUG
    [[nodiscard]] debug_stats get_debug_stats() const noexcept {
        return debug_stats{
            size_,
            capacity_,
            rehash_count_,
            double_rehash_count_,
            max_probe_len_,
            probe_ops_ == 0 ? 0.0 : static_cast<double>(total_probes_) / static_cast<double>(probe_ops_)};
    }
#endif

private:
    static constexpr key_type k_unoccupied = std::numeric_limits<UInt32>::max();

    hasher_type hasher_{};
    allocator_type alloc_{};
    slot_allocator_type slot_alloc_{};
    value_allocator_type value_alloc_{};

    slot_type *slots_ = nullptr; // 只包含 key
    value_type *values_ = nullptr;

    size_type capacity_ = 0;
    size_type bucket_mask_ = 0;
    size_type size_ = 0;
    size_type growth_limit_ = 0;
    float max_load_factor_ = detail::k_default_max_load_factor;

#ifdef DEBUG
    mutable SizeT total_probes_ = 0;
    mutable SizeT probe_ops_ = 0;
    mutable SizeT max_probe_len_ = 0;
    mutable SizeT rehash_count_ = 0;
    mutable SizeT double_rehash_count_ = 0;

    void record_probe(SizeT probe_len) const noexcept {
        total_probes_ += probe_len;
        ++probe_ops_;
        if (probe_len > max_probe_len_) max_probe_len_ = probe_len;
    }
#endif

    SizeT hash_of(const key_type &key) const noexcept { return hasher_(key); }

    template <InsertPolicy Policy> auto make_result(size_type index, bool inserted) -> insert_return_type<Policy> {
        if constexpr (Policy::return_value) {
            if constexpr (Policy::check_dup) {
                return {iterator{this, index}, inserted};
            } else {
                return iterator{this, index};
            }
        } else {
            return;
        }
    }

    // 如果要 find hit 快，先判断是否等于 key 再判是否为未占用
    // 如果要 find miss 快，则反过来
    // 当前这个版本，对于 find hit 最为友好，谨慎调整
    // 返回 capacity_ 代替 npos，减少外部分支
    size_type index_of(const key_type &key) const noexcept {
        size_type index = static_cast<size_type>(hash_of(key)) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            key_type stored = slots_[index].key;
            if (stored == key) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            if (stored == k_unoccupied) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return capacity_; // 用 capacity_ 代替 npos，减少外部分支
            }
            index = (index + 1) & bucket_mask_;
        }
    }

    size_type find_or_insert_default(const key_type &key) {
        if (size_ >= growth_limit_) { // 临界元素
            size_type existing = index_of(key);
            if (existing != capacity_) return existing;
            double_storage();
        }
        size_type index = static_cast<size_type>(hash_of(key)) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            slot_type &slot = slots_[index];
            key_type stored_key = slot.key;
            if (stored_key == k_unoccupied) {
                slot.key = key;
                std::construct_at(values_ + index, std::piecewise_construct, std::forward_as_tuple(key),
                                  std::forward_as_tuple());
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            if (stored_key == key) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            index = (index + 1) & bucket_mask_;
        }
    }

    template <typename M>
        requires(std::constructible_from<mapped_type, M>)
    std::pair<size_type, bool> find_and_insert_or_assign(const key_type &key, M &&mapped) {
        if (size_ >= growth_limit_) {
            size_type existing = index_of(key);
            if (existing != capacity_) {
                values_[existing].second = std::forward<M>(mapped);
                return {existing, false};
            }
            double_storage();
        }
        size_type index = static_cast<size_type>(hash_of(key)) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            slot_type &slot = slots_[index];
            key_type stored_key = slot.key;
            if (stored_key == k_unoccupied) {
                slot.key = key;
                std::construct_at(values_ + index, std::piecewise_construct, std::forward_as_tuple(key),
                                  std::forward_as_tuple(std::forward<M>(mapped)));
                ++size_;
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return {index, true};
            }
            if (stored_key == key) {
                values_[index].second = std::forward<M>(mapped);
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return {index, false};
            }
            index = (index + 1) & bucket_mask_;
        }
    }

    void allocate_storage(size_type capacity) {
        slots_ = slot_alloc_traits::allocate(slot_alloc_, capacity);
        values_ = value_alloc_traits::allocate(value_alloc_, capacity);
        capacity_ = capacity;
        bucket_mask_ = capacity_ - 1;
        growth_limit_ = static_cast<size_type>(max_load_factor_ * static_cast<float>(capacity_));

        for (size_type index = 0; index < capacity_; ++index) {
            std::construct_at(slots_ + index); // 默认 key = k empty key
        }
    }

    void deallocate_storage() noexcept {
        slot_alloc_traits::deallocate(slot_alloc_, slots_, capacity_);
        value_alloc_traits::deallocate(value_alloc_, values_, capacity_);
    }

    // 注意这里 destroy all 和泛型版本不一样，每个槽位都有构造过的 Slot 对象
    // TODO：在 valuetype 也是 pod 类型的情况下，可以直接省去 destroy 的步骤
    void destroy_all() noexcept {
        for (size_type index = 0; index < capacity_; ++index) {
            if (slots_[index].key != k_unoccupied) std::destroy_at(values_ + index);
            std::destroy_at(slots_ + index);
        }
    }

    void move_old_elements(slot_type *old_slots, value_type *old_values, size_type old_capacity) {
        for (size_type index = 0; index < old_capacity; ++index) {
            slot_type &old_slot = old_slots[index];
            key_type old_key = old_slot.key;
            if (old_slot.key == k_unoccupied) {
                std::destroy_at(old_slots + index); // 析构 Slot
                continue;
            }
            size_type insert_index = find_insert_index_for_rehash(old_key);
            // allocate storage 已经构造好了 slot 对象
            slot_type &new_slot = slots_[insert_index];
            new_slot.key = old_key;

            std::construct_at(values_ + insert_index, std::move(old_values[index]));
            std::destroy_at(old_slots + index);
            std::destroy_at(old_values + index);
        }
        slot_alloc_traits::deallocate(slot_alloc_, old_slots, old_capacity);
        value_alloc_traits::deallocate(value_alloc_, old_values, old_capacity);
    }

    void double_storage() {
        slot_type *old_slots = slots_;
        value_type *old_values = values_;
        size_type old_capacity = capacity_;
        allocate_storage(capacity_ * 2);
#ifdef DEBUG
        ++double_rehash_count_;
        ++rehash_count_;
#endif
        move_old_elements(old_slots, old_values, old_capacity);
    }

    size_type find_insert_index_for_rehash(const key_type &key) {
        size_type index = static_cast<size_type>(hash_of(key)) & bucket_mask_;
#ifdef DEBUG
        SizeT probe_len = 0;
#endif
        for (;;) {
#ifdef DEBUG
            ++probe_len;
#endif
            if (slots_[index].key == k_unoccupied) {
#ifdef DEBUG
                record_probe(probe_len);
#endif
                return index;
            }
            index = (index + 1) & bucket_mask_;
        }
    }

    void erase_at_index(size_type index) noexcept {
        // 左移 cluster
        size_type cur = index;
        for (;;) {
            size_type next = (cur + 1) & bucket_mask_;
            slot_type &next_slot = slots_[next];
            key_type next_key = next_slot.key;
            if (next_key == k_unoccupied) break; // 遇到 EMPTY

            SizeT next_hash = hash_of(next_key);
            size_type home = static_cast<size_type>(next_hash) & bucket_mask_;
            if (home == next) break; // 下一个元素在自己家上，不要左移

            slots_[cur].key = next_key;
            values_[cur] = std::move(values_[next]);
            cur = next;
        }
        slots_[cur].key = k_unoccupied;
        std::destroy_at(values_ + cur);
        --size_;
    }

    friend bool operator==(const flat_hash_map_u32 &lhs, const flat_hash_map_u32 &rhs) {
        if (&lhs == &rhs) return true;
        if (lhs.size_ != rhs.size_) return false;
        if (lhs.size_ == 0) return true;
        for (size_type index = 0; index < lhs.capacity_; ++index) {
            const slot_type &slot = lhs.slots_[index];
            key_type stored_key = slot.key;
            if (stored_key == k_unoccupied) continue;

            const value_type &kv = lhs.values_[index];
            size_type result = rhs.index_of(kv.first);
            if (result == rhs.capacity_) return false;
            if (!(kv.second == rhs.values_[result].second)) return false;
        }
        return true;
    }

    friend void swap(flat_hash_map_u32 &lhs, flat_hash_map_u32 &rhs) noexcept(noexcept(lhs.swap(rhs))) {
        lhs.swap(rhs);
    }
}; // flat hash map uint32
} // namespace mcl