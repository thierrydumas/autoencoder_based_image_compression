cmake_minimum_required(VERSION 2.8)
project(lossless)
file(GLOB CPP_FILES ../source/*.cpp)
file(GLOB H_FILES ../source/*.h)
set(SOURCE_FILES ${CPP_FILES} ${H_FILES})
add_executable(executable ${SOURCE_FILES})
target_compile_options(executable PUBLIC -std=c++1y -Wall -Wfloat-conversion)


