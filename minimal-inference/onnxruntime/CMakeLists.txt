cmake_minimum_required(VERSION 3.16)
project(minimal-onnxruntime C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-onnxruntime
  minimal-onnxruntime.cpp
)

target_link_libraries(minimal-onnxruntime
  onnxruntime
)

if (MSVC)
    file(GLOB_RECURSE ONNX_DLLS "${CMAKE_SOURCE_DIR}/modules/onnx*/*.dll")
    add_custom_command(TARGET minimal-onnxruntime
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNX_DLLS}
            $<TARGET_FILE_DIR:minimal-onnxruntime>)
endif (MSVC)