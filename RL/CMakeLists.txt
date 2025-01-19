cmake_minimum_required(VERSION 3.10)
project(ReforceLearning)
set(CMAKE_CXX_STANDARD 11)

#find_package(GTEST REQUIRED)
include_directories(CliffWalkingEnv)
include_directories(DataType)
include_directories(PolicyIteration)

# 查找 gtest 库
find_package(GTest REQUIRED)

# 包含 gtest 的头文件目录
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(ReforceLearning ReforceLearning.cpp 
               CliffWalkingEnv/CWalkEnv.cpp
               PolicyIteration/PolicyIteration.cpp)

               # 链接 gtest 库
target_link_libraries(ReforceLearning ${GTEST_LIBRARIES} pthread)
