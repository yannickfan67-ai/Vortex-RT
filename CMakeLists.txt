cmake_minimum_required(VERSION 3.24)

project(VortexRT VERSION 0.2.0 LANGUAGES CXX)

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
set(Q8_MATVEC_SPV "${VORTEXRT_SHADER_DIR}/q8_matvec.comp.spv")
set(Q8_MATVEC_U8_SPV "${VORTEXRT_SHADER_DIR}/q8_matvec_u8.comp.spv")
set(Q8_MATVEC_32_SPV "${VORTEXRT_SHADER_DIR}/q8_matvec_32.comp.spv")
set(Q8_MATVEC_U8_32_SPV "${VORTEXRT_SHADER_DIR}/q8_matvec_u8_32.comp.spv")
set(ARGMAX_SPV "${VORTEXRT_SHADER_DIR}/argmax.comp.spv")
set(GELU_MUL_SPV "${VORTEXRT_SHADER_DIR}/gelu_mul.comp.spv")

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

add_custom_command(
    OUTPUT "${Q8_MATVEC_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VORTEXRT_SHADER_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec.comp"
            -o "${Q8_MATVEC_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec.comp"
    VERBATIM
)

add_custom_command(
    OUTPUT "${Q8_MATVEC_U8_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VORTEXRT_SHADER_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec_u8.comp"
            -o "${Q8_MATVEC_U8_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec_u8.comp"
    VERBATIM
)

add_custom_command(
    OUTPUT "${ARGMAX_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VORTEXRT_SHADER_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/argmax.comp"
            -o "${ARGMAX_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/argmax.comp"
    VERBATIM
)

add_custom_command(
    OUTPUT "${GELU_MUL_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VORTEXRT_SHADER_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gelu_mul.comp"
            -o "${GELU_MUL_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gelu_mul.comp"
    VERBATIM
)

add_custom_command(
    OUTPUT "${Q8_MATVEC_32_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VORTEXRT_SHADER_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec_32.comp"
            -o "${Q8_MATVEC_32_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec_32.comp"
    VERBATIM
)

add_custom_command(
    OUTPUT "${Q8_MATVEC_U8_32_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VORTEXRT_SHADER_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec_u8_32.comp"
            -o "${Q8_MATVEC_U8_32_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_matvec_u8_32.comp"
    VERBATIM
)

add_custom_target(
    vortexrt_shaders ALL
    DEPENDS
        "${VECTOR_ADD_SPV}"
        "${Q8_MATVEC_SPV}"
        "${Q8_MATVEC_U8_SPV}"
        "${Q8_MATVEC_32_SPV}"
        "${Q8_MATVEC_U8_32_SPV}"
        "${ARGMAX_SPV}"
        "${GELU_MUL_SPV}"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/vortexrt/build_config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/vortexrt/build_config.hpp"
    @ONLY
)

add_library(vortexrt STATIC
    src/vulkan_context.cpp
    src/buffer.cpp
    src/compute_pipeline.cpp
    src/q8_matvec.cpp
    src/argmax.cpp
    src/gguf.cpp
    src/gemma3.cpp
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

add_executable(vortexrt-q8-test src/q8_test_main.cpp)
target_link_libraries(vortexrt-q8-test PRIVATE vortexrt)
add_dependencies(vortexrt-q8-test vortexrt_shaders)

add_executable(vortexrt-gguf-info src/gguf_main.cpp)
target_link_libraries(vortexrt-gguf-info PRIVATE vortexrt)

add_executable(vortexrt-gemma src/gemma_main.cpp)
target_link_libraries(vortexrt-gemma PRIVATE vortexrt)
add_dependencies(vortexrt-gemma vortexrt_shaders)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME vortexrt-info COMMAND vortexrt-info)
    add_test(
        NAME vortexrt-bench-smoke
        COMMAND vortexrt-bench --elements 65536 --repetitions 2
    )
    add_test(NAME vortexrt-q8-matvec COMMAND vortexrt-q8-test)
endif()
