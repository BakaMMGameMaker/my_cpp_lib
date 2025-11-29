#pragma once
#include <functional>
#include <queue>
#include <vector>

template <typename T> class SFastMedian {

    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");

    std::priority_queue<T, std::vector<T>, std::less<T>> left_heap;
    std::priority_queue<T, std::vector<T>, std::greater<T>> right_heap;

    void rebalance() {
        if (left_heap.size() > right_heap.size() + 1) {
            right_heap.push(left_heap.top());
            left_heap.pop();
        } else if (right_heap.size() > left_heap.size()) {
            left_heap.push(right_heap.top());
            right_heap.pop();
        }
    }

public:
    void push(const T &value) {
        if (left_heap.empty() || value <= left_heap.top()) left_heap.push(value);
        else right_heap.push(value);
        rebalance();
    }

    // return 0.0 if no data
    [[nodiscard]] double get() const noexcept {
        if (left_heap.empty()) [[unlikely]]
            return 0.0;
        if (left_heap.size() == right_heap.size()) return (left_heap.top() + right_heap.top()) * 0.5;
        return static_cast<double>(left_heap.top());
    }
};