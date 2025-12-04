# STrie – 高性能前缀树 (C++20/23)

一个面向游戏开发与性能敏感场景的轻量级、高性能、可定制前缀树实现。
相比普通的 trie，本项目支持动态/稠密混合存储、可选节点复用、无异常压力、并且为高频查找优化。

## ✨ 特性

* **高性能**：缓存友好，节点布局紧凑，适合频繁前缀查询。
* **可定制子节点存储**：通过 `ChildrenStorageType` 决定使用稠密数组、稀疏数组和动态稠密数组混合策略，允许自由调整扩容阈值。
* **可选节点复用**：可复用删掉的节点以减少内存碎片 `ReuseDeadNodes = true/false`。
* **模板化 key/value 类型**：支持 `void` 作为无值模式（只存终止标记）。
* **线程安全读取（可加 shared_mutex）**：适合游戏查询场景。
* **零依赖**：仅 `<string_view>`, `<vector>` 等标准库。

## 📁 目录结构

```
my_cpp_lib/
  stries/
    strie.hpp                  // 核心 Trie 实现
    stries_children_storage.hpp// 子节点存储策略（Fixed、Hybrid 等）
    stries_free_list.hpp       // 可选节点复用逻辑
    test.cpp                   // 使用示例与基本测试
```

## 🚀 快速上手

### 引入头文件

```cpp
#include "stries/strie.hpp"
#include "stries/stries_children_storage.hpp"
```

### 定义一个简单的 Trie

```cpp
// 这里使用 HybridDynamicChildren，并且存储 int 类型的值
using MyTrie = STrie<HybridDynamicChildren<UInt32, 256, 16>, true, int>;

MyTrie trie;
trie.insert_or_assign("apple", 1);
trie.insert_or_assign("app", 2);

int* p = trie.find("app");
if (p) {
    std::cout << *p << "\n"; // 输出 2
}
```

### 遍历所有单词

```cpp
for (auto it = trie.begin(); it != trie.end(); ++it) {
    auto [word, value] = *it;
    std::cout << word << ": " << value << "\n";
}
```

### 基于前缀搜索

```cpp
trie.for_each_with_prefix("ab", [](std::string_view word, auto& value){
    value += 10;
});
```

## 🧪 测试

运行测试：

```bash
clang++ -std=c++23 -O2 test.cpp -I./ -o test.exe
./test.exe
```

## 🧱 设计亮点

### 1. Node 结构清晰

每个节点包含：

* `children`：由模板参数控制的子节点容器策略
* `value` 或 `is_end`：根据 `ValueType` 是否为 `void` 决定
* `reset_node / reset_value`：方便回收与重建

### 2. 强类型 Index + 连续 node_pool

避免指针失效，提高整体 cache locality。

### 3. 两类迭代器

* `iterator`
* `const_iterator`

DFS 栈结构保证按字典序遍历。

## 🛠 构建环境

* **Windows**
* **Clang/LLVM 21+**
* **C++20/C++23**
* VSCode + clangd

## 📌 TODO（根据你的项目可继续扩展）

* prefix filter API
* value mutation with custom callback
* iterator category 完善
* 高阶 score 函数
* 简易序列化（已有 k_magic/k_version）

## 📜 License

MIT