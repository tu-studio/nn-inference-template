cmake_minimum_required(VERSION 3.16)
project(validate-real-time-principles-onnxruntime C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(validate-real-time-principles-onnxruntime
  validate-real-time-principles-onnxruntime.cpp
)

target_compile_options(validate-real-time-principles-onnxruntime PUBLIC -fsanitize=realtime)
target_link_options(validate-real-time-principles-onnxruntime PUBLIC -fsanitize=realtime)

target_link_libraries(validate-real-time-principles-onnxruntime
  onnxruntime
)

if (MSVC)
    file(GLOB_RECURSE ONNX_DLLS "${CMAKE_SOURCE_DIR}/modules/onnx*/*.dll")
    add_custom_command(TARGET validate-real-time-principles-onnxruntime
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNX_DLLS}
            $<TARGET_FILE_DIR:validate-real-time-principles-onnxruntime>)
endif (MSVC)