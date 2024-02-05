cmake_minimum_required(VERSION 3.16)
project(minimal-libtorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-libtorch
  minimal-libtorch.cpp
)

target_link_libraries(minimal-libtorch
  aari::aari
)

if (MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        file(GLOB_RECURSE LIBTORCH_DLLS "${CMAKE_SOURCE_DIR}/modules/libtorch*debug/*.dll")
    else()
        file(GLOB_RECURSE LIBTORCH_DLLS "${CMAKE_SOURCE_DIR}/modules/libtorch*release/*.dll")
    endif()
    add_custom_command(TARGET minimal-libtorch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${LIBTORCH_DLLS}
            $<TARGET_FILE_DIR:minimal-libtorch>)
endif (MSVC)

