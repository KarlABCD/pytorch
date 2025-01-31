cmake_minimum_required(VERSION 3.10)
project(ReforceLearning)
set(CMAKE_CXX_STANDARD 11)

# 查找 gtest 库
find_package(GTest REQUIRED)

# 包含 gtest 的头文件目录
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(CliffWalkingEnv DataType PolicyIteration VectorCalc 
                    StringOper QLearning)

# 添加可执行文件
add_executable(ReforceLearning ReforceLearning.cpp 
               CliffWalkingEnv/CWalkEnv.cpp
               PolicyIteration/PolicyIteration.cpp
               VectorCalc/VectorCalc.cpp
               StringOper/StringOper.cpp
               QLearning/QLearning.cpp)

# 链接 gtest 库
target_link_libraries(ReforceLearning ${GTEST_LIBRARIES} pthread)

# 启用测试
enable_testing()

# 添加测试
add_test(NAME VectorCalcTest COMMAND VectorCalcTest)
