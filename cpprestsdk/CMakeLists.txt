cmake_minimum_required(VERSION 3.6)
project("server")

set(Torch_DIR ./libtorch/share/cmake/Torch)

find_package(PkgConfig)
find_package(Torch REQUIRED)
pkg_check_modules(Pistache REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PkgConfig::Pistache)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${CMAKE_THREAD_LIBS_INIT})