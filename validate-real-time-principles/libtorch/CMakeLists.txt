cmake_minimum_required(VERSION 3.16)
project(validate-real-time-principles-libtorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(validate-real-time-principles-libtorch
  validate-real-time-principles-libtorch.cpp
)

target_compile_options(validate-real-time-principles-libtorch PUBLIC -fsanitize=realtime)
target_link_options(validate-real-time-principles-libtorch PUBLIC -fsanitize=realtime)

target_link_libraries(validate-real-time-principles-libtorch
  torch
)

if (MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        file(GLOB_RECURSE LIBTORCH_DLLS "${CMAKE_SOURCE_DIR}/modules/libtorch*debug/*.dll")
    else()
        file(GLOB_RECURSE LIBTORCH_DLLS "${CMAKE_SOURCE_DIR}/modules/libtorch*release/*.dll")
    endif()
    add_custom_command(TARGET validate-real-time-principles-libtorch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${LIBTORCH_DLLS}
            $<TARGET_FILE_DIR:validate-real-time-principles-libtorch>)
endif (MSVC)

