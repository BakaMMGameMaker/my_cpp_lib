# 高性能游戏想 flat hash map
1. 
    UB：it->first 改 key
        iterator it; ++it
    erase(it from other map)
    erase(range from other map)