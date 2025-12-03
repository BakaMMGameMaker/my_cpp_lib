#include "sflatmap.hpp"
#include <cassert>
#include <iostream>

int main() {
    sflat::flat_hash_map<int, int> mp; // 注意默认构造，capacity 为 0
    std::cout << "map created" << std::endl;

    mp.insert({1, 10});
    std::cout << "insert 1, 10" << std::endl;

    mp.insert_or_assign(2, 20);
    std::cout << "insert or assign 2 20" << std::endl;

    mp[3] = 30;
    std::cout << "map[3] = 30" << std::endl;

    assert(mp.size() == 3);
    std::cout << "assert mp.size == 3" << std::endl;

    assert(mp.at(1) == 10);
    std::cout << "assert mp.at 1 == 10" << std::endl;

    assert(mp.contains(2));
    std::cout << "assert mp.contains 2" << std::endl;

    mp.erase(2);
    std::cout << "erase 2" << std::endl;

    assert(!mp.contains(2));
    std::cout << "assert !mp.contains 2" << std::endl;

    for (auto it = mp.begin(); it != mp.end(); ++it) {
        if (it->first == 1) it->second = 114514;
    }

    assert(mp[1] == 114514);
    std::cout << "assert mp[1] == 114514" << std::endl;

    std::cout << "All tests passed" << std::endl;
}