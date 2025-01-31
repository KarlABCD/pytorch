#!/bin/bash

# 编译程序
cmake --build build

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败"
    exit 1
fi

EXECUTABLE_FILE="ReforceLearning.exe"
BUILD_DIR="build"

# 检查 VS Code 是否成功启动调试
if [ $? -eq 0 ]; then
    echo "编译成功"
    #code -g C++ Debug
fi