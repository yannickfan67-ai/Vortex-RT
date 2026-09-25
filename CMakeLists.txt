cmake_minimum_required(VERSION 3.24)

project(VortexRT VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan 1.3 REQUIRED)

find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK and ensure glslc is on PATH.")
endif()

set(VORTEXRT_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(VECTOR_ADD_SPV "${VORTEXRT_SHADER_DIR}/vector_add.comp.spv")

add_custom_command(
    OUTPUT "${VECTOR_ADD_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VORTEXRT_SHADER_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vector_add.comp"
            -o "${VECTOR_ADD_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vector_add.comp"
    VERBATIM
)

add_custom_target(vortexrt_shaders ALL DEPENDS "${VECTOR_ADD_SPV}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/vortexrt/build_config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/vortexrt/build_config.hpp"
    @ONLY
)

add_library(vortexrt STATIC
    src/vulkan_context.cpp
    src/buffer.cpp
    src/compute_pipeline.cpp
    src/gguf.cpp
)

target_include_directories(vortexrt
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(vortexrt PUBLIC Vulkan::Vulkan)

if(MSVC)
    target_compile_options(vortexrt PRIVATE /W4 /permissive-)
else()
    target_compile_options(vortexrt PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(vortexrt-info src/info_main.cpp)
target_link_libraries(vortexrt-info PRIVATE vortexrt)

add_executable(vortexrt-bench src/bench_main.cpp)
target_link_libraries(vortexrt-bench PRIVATE vortexrt)
add_dependencies(vortexrt-bench vortexrt_shaders)

add_executable(vortexrt-gguf-info src/gguf_main.cpp)
target_link_libraries(vortexrt-gguf-info PRIVATE vortexrt)
