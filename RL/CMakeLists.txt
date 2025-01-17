cmake_minimum_required(VERSION 3.10)
project(ReforceLearning)
set(CMAKE_CXX_STANDARD 11)

#find_package(GTEST REQUIRED)
include_directories(CliffWalkingEnv)
include_directories(DataType)
include_directories(PolicyIteration)
add_executable(ReforceLearning ReforceLearning.cpp 
               CliffWalkingEnv/CWalkEnv.cpp
               PolicyIteration/PolicyIteration.cpp)