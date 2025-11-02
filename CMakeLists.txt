cmake_minimum_required(VERSION 4.0)
project(Vulkan)

include_directories("$ENV{VULKAN_SDK}/include")
link_directories("$ENV{VULKAN_SDK}/lib")

find_package(Slang CONFIG HINTS "$ENV{VULKAN_SDK}/lib/cmake")

set(CMAKE_CXX_STANDARD 20)

add_executable(Vulkan main.cpp)
target_compile_definitions(Vulkan PRIVATE VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)

target_link_libraries(Vulkan glfw)