//
// Created by Valentin Ackva on 03/02/2024.
//

#ifndef NN_INFERENCE_TEMPLATE_INFERENCECONFIGNEW_H
#define NN_INFERENCE_TEMPLATE_INFERENCECONFIGNEW_H

#include <array>
#include <string>
#include <vector>

struct InferenceConfig {
    InferenceConfig(
#ifdef USE_LIBTORCH
            const std::string model_path_torch,
            const std::vector<int> model_input_shape_torch,
            const std::vector<int> model_output_shape_torch,
            const int model_latency_torch,
#endif
#ifdef USE_ONNXRUNTIME
            const std::string model_path_onnx,
            const std::vector<int> model_input_shape_onnx,
            const std::vector<int> model_output_shape_onnx,
            const int model_latency_onnx,
#endif
#ifdef USE_TFLITE
            const std::string model_path_tflite,
            const std::vector<int> model_input_shape_tflite,
            const std::vector<int> model_output_shape_tflite,
            const int model_latency_tflite,
#endif
            int batch_size,
            int model_input_size,
            int model_input_size_backend,
            int model_output_size_backend,
            int max_inference_time) :
#ifdef USE_LIBTORCH
            m_model_path_torch(model_path_torch),
            m_model_input_shape_torch(model_input_shape_torch),
            m_model_output_shape_torch(model_output_shape_torch),
            m_model_latency_torch(model_latency_torch),
#endif
#ifdef USE_ONNXRUNTIME
            m_model_path_onnx(model_path_onnx),
            m_model_input_shape_onnx(model_input_shape_onnx),
            m_model_output_shape_onnx(model_output_shape_onnx),
            m_model_latency_onnx(model_latency_onnx),
#endif
#ifdef USE_TFLITE
            m_model_path_tflite(model_path_tflite),
            m_model_input_shape_tflite(model_input_shape_tflite),
            m_model_output_shape_tflite(model_output_shape_tflite),
            m_model_latency_tflite(model_latency_tflite),
#endif
            m_batch_size(batch_size),
            m_model_input_size(model_input_size),
            m_model_input_size_backend(model_input_size_backend),
            m_model_output_size_backend(model_output_size_backend),
            m_max_inference_time(max_inference_time)
    {}

    const int m_batch_size;
    const int m_model_input_size;
    const int m_model_input_size_backend;
    const int m_model_output_size_backend;
    const int m_max_inference_time;

#ifdef USE_LIBTORCH
    const std::string m_model_path_torch;
    const std::vector<int> m_model_input_shape_torch;
    const std::vector<int> m_model_output_shape_torch;
    const int m_model_latency_torch;
#endif

#ifdef USE_ONNXRUNTIME
    const std::string m_model_path_onnx;
    const std::vector<int> m_model_input_shape_onnx;
    const std::vector<int> m_model_output_shape_onnx;
    const int m_model_latency_onnx;
#endif

#ifdef USE_TFLITE
    const std::string m_model_path_tflite;
    const std::vector<int> m_model_input_shape_tflite;
    const std::vector<int> m_model_output_shape_tflite;
    const int m_model_latency_tflite;
#endif
};


#endif //NN_INFERENCE_TEMPLATE_INFERENCECONFIGNEW_H
