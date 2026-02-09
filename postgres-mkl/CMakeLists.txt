cmake_minimum_required(VERSION 3.13)

project(postgre-mkl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MKL CONFIG REQUIRED)
message(STATUS "${MKL_IMPORTED_TARGETS}")

add_library(matmul SHARED matmul.c)
target_compile_options(matmul PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(matmul PUBLIC
    /usr/include/postgresql/12/server
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(matmul PUBLIC $<LINK_ONLY:MKL::MKL>)

add_library(einsql SHARED einsql.c)
target_compile_options(einsql PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(einsql PUBLIC
    /usr/include/postgresql/12/server
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(einsql PUBLIC $<LINK_ONLY:MKL::MKL>)
