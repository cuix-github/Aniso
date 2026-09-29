cmake_minimum_required(VERSION 3.20)
project(Aniso LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(aniso_core
    src/ply_loader.cpp
    src/camera.cpp
    src/image.cpp
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
