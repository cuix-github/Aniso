cmake_minimum_required(VERSION 3.20)
project(Aniso LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/utf-8>")
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
endif()

# The CUDA renderer is optional: build_cuda.bat turns it on with the project-local toolkit.
option(ANISO_CUDA "Build the CUDA renderer" OFF)

add_library(aniso_core
    src/ply_loader.cpp
    src/camera.cpp
    src/image.cpp
    src/render.cpp
    src/sh.cpp
)
target_include_directories(aniso_core PUBLIC include)

add_executable(aniso src/main.cpp)
target_link_libraries(aniso PRIVATE aniso_core)

enable_testing()
add_executable(test_ply_loader tests/test_ply_loader.cpp)
target_link_libraries(test_ply_loader PRIVATE aniso_core)
add_test(NAME ply_loader COMMAND test_ply_loader)

add_executable(test_camera tests/test_camera.cpp)
target_link_libraries(test_camera PRIVATE aniso_core)
add_test(NAME camera COMMAND test_camera)

add_executable(test_render tests/test_render.cpp)
target_link_libraries(test_render PRIVATE aniso_core)
add_test(NAME render COMMAND test_render)

add_executable(test_sh tests/test_sh.cpp)
target_link_libraries(test_sh PRIVATE aniso_core)
add_test(NAME sh COMMAND test_sh)

if(WIN32)
    add_executable(aniso_viewer src/viewer.cpp)
    target_link_libraries(aniso_viewer PRIVATE aniso_core user32 gdi32)
endif()

if(ANISO_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 120)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    add_library(aniso_gpu src/gpu_render.cu)
    target_link_libraries(aniso_gpu PUBLIC aniso_core)
    target_compile_options(aniso_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>")
    target_compile_definitions(aniso PRIVATE ANISO_WITH_CUDA)
    target_link_libraries(aniso PRIVATE aniso_gpu)
    if(WIN32)
        target_compile_definitions(aniso_viewer PRIVATE ANISO_WITH_CUDA)
        target_link_libraries(aniso_viewer PRIVATE aniso_gpu)
    endif()
    add_executable(test_gpu_render tests/test_gpu_render.cpp)
    target_link_libraries(test_gpu_render PRIVATE aniso_gpu)
    add_test(NAME gpu_render COMMAND test_gpu_render)
endif()
