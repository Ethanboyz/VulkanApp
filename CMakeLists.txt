cmake_minimum_required(VERSION 4.0)
project(Vulkan)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_CLANG_TIDY; clang-tidy;)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "VULKAN_SDK = '$ENV{VULKAN_SDK}'")

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(SYSTEM "$ENV{VULKAN_SDK}/include")
link_directories("$ENV{VULKAN_SDK}/lib")

find_package(Slang CONFIG HINTS "$ENV{VULKAN_SDK}/lib/cmake")

add_executable(Vulkan src/main.cpp)
target_compile_definitions(Vulkan PRIVATE VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)
target_compile_options(Vulkan PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -Werror
    -Wno-missing-field-initializers
)

target_link_libraries(Vulkan PRIVATE Vulkan::Vulkan glfw)

# Find the slang compiler
find_program(SLANGC_EXECUTABLE
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/bin"
)
if (NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found. Set SLANGC_EXECUTABLE or install Slang.")
endif()

function(add_slang_shader_target TARGET)
    cmake_parse_arguments(SHADER "" "" "SOURCES" ${ARGN})

    set(SHADERS_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(SHADERS_BIN_DIR  ${CMAKE_CURRENT_BINARY_DIR}/shaders)

    set(ENTRY_POINTS -entry vertex_main -entry fragment_main)

    add_custom_command(
        OUTPUT ${SHADERS_BIN_DIR}/slang.spv
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_BIN_DIR}
        COMMAND ${CMAKE_COMMAND} -E env
        "LD_LIBRARY_PATH=$ENV{VULKAN_SDK}/lib:$ENV{LD_LIBRARY_PATH}"
        ${SLANGC_EXECUTABLE}
        ${SHADER_SOURCES}
        -target spirv
        -profile spirv_1_4
        -emit-spirv-directly
        -fvk-use-entrypoint-name
        ${ENTRY_POINTS}
        -o ${SHADERS_BIN_DIR}/slang.spv
        WORKING_DIRECTORY ${SHADERS_SRC_DIR}
        DEPENDS ${SHADER_SOURCES}
        COMMENT "Compiling Slang Shaders..."
        VERBATIM
    )

    add_custom_target(${TARGET}
            DEPENDS ${SHADERS_BIN_DIR}/slang.spv
    )
endfunction()
set(SHADER_SLANG_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
add_slang_shader_target(shaders SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(Vulkan shaders)