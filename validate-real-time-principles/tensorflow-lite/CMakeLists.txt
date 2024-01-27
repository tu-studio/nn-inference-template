cmake_minimum_required(VERSION 3.16)
project(validate-real-time-principles-tflite C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(validate-real-time-principles-tflite
  validate-real-time-principles-tflite.cpp
)

target_compile_options(validate-real-time-principles-tflite PUBLIC -fsanitize=realtime)
target_link_options(validate-real-time-principles-tflite PUBLIC -fsanitize=realtime)

target_link_libraries(validate-real-time-principles-tflite
  tensorflowlite_c
)

if (MSVC)
    file(GLOB_RECURSE TFLITE_DLLS "${CMAKE_SOURCE_DIR}/modules/tensorflowlite*/*.dll")
    add_custom_command(TARGET validate-real-time-principles-tflite
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TFLITE_DLLS}
            $<TARGET_FILE_DIR:validate-real-time-principles-tflite>)
endif (MSVC)