#include "sflatmap.hpp"
#include <cassert>
#include <iostream>

int main() {
    sflat::flat_hash_map<int, int> mp; // 注意默认构造，capacity 为 0

    mp.insert({1, 10});

    mp.insert_or_assign(2, 20);

    mp[3] = 30;

    assert(mp.size() == 3);

    assert(mp.at(1) == 10);

    assert(mp.contains(2));

    mp.erase(2);

    assert(!mp.contains(2));

    for (auto it = mp.begin(); it != mp.end(); ++it) {
        if (it->first == 1) it->second = 114514;
    }

    assert(mp[1] == 114514);

    std::cout << "All tests passed" << std::endl;
}