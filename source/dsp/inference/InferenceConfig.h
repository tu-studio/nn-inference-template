#ifndef NN_INFERENCE_TEMPLATE_INFERENCECONFIG_H
#define NN_INFERENCE_TEMPLATE_INFERENCECONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>

struct InferenceConfig {
    InferenceConfig(
#ifdef USE_LIBTORCH
            const std::string model_path_torch,
            const std::vector<int64_t> model_input_shape_torch,
            const std::vector<int64_t> model_output_shape_torch,
            const int model_latency_torch,
            const bool model_stateful_torch,
#endif
#ifdef USE_ONNXRUNTIME
            const std::string model_path_onnx,
            const std::vector<int64_t> model_input_shape_onnx,
            const std::vector<int64_t> model_output_shape_onnx,
            const int model_latency_onnx,
            const bool model_stateful_onnx,
#endif
#ifdef USE_TFLITE
            const std::string model_path_tflite,
            const std::vector<int64_t> model_input_shape_tflite,
            const std::vector<int64_t> model_output_shape_tflite,
            const int model_latency_tflite,
            const bool model_stateful_tflite,
#endif
            int batch_size,
            int model_input_size,
            int model_input_size_backend,
            int model_output_size_backend,
            int max_inference_time,
            int model_latency,
            bool warm_up = true,
            int numberOfThreads = std::thread::hardware_concurrency() - 1) :
#ifdef USE_LIBTORCH
            m_model_path_torch(model_path_torch),
            m_model_input_shape_torch(model_input_shape_torch),
            m_model_output_shape_torch(model_output_shape_torch),
            m_model_latency_torch(model_latency_torch),
            m_model_stateful_torch(model_stateful_torch),
#endif
#ifdef USE_ONNXRUNTIME
            m_model_path_onnx(model_path_onnx),
            m_model_input_shape_onnx(model_input_shape_onnx),
            m_model_output_shape_onnx(model_output_shape_onnx),
            m_model_latency_onnx(model_latency_onnx),
            m_model_stateful_onnx(model_stateful_onnx),
#endif
#ifdef USE_TFLITE
            m_model_path_tflite(model_path_tflite),
            m_model_input_shape_tflite(model_input_shape_tflite),
            m_model_output_shape_tflite(model_output_shape_tflite),
            m_model_latency_tflite(model_latency_tflite),
            m_model_stateful_tflite(model_stateful_tflite),
#endif
            m_batch_size(batch_size),
            m_model_input_size(model_input_size),
            m_model_input_size_backend(model_input_size_backend),
            m_model_output_size_backend(model_output_size_backend),
            m_max_inference_time(max_inference_time),
            m_model_latency(model_latency),
            m_warm_up(warm_up),
            m_number_of_threads(numberOfThreads)
    {}

    const int m_batch_size;
    const int m_model_input_size;
    const int m_model_input_size_backend;
    const int m_model_output_size_backend;
    const int m_max_inference_time;
    const int m_model_latency;
    const bool m_warm_up;

    const int m_number_of_threads;

#ifdef USE_LIBTORCH
    const std::string m_model_path_torch;
    const std::vector<int64_t> m_model_input_shape_torch;
    const std::vector<int64_t> m_model_output_shape_torch;
    const int m_model_latency_torch;
    const bool m_model_stateful_torch;
#endif

#ifdef USE_ONNXRUNTIME
    const std::string m_model_path_onnx;
    const std::vector<int64_t> m_model_input_shape_onnx;
    const std::vector<int64_t> m_model_output_shape_onnx;
    const int m_model_latency_onnx;
    const bool m_model_stateful_onnx;
#endif

#ifdef USE_TFLITE
    const std::string m_model_path_tflite;
    const std::vector<int64_t> m_model_input_shape_tflite;
    const std::vector<int64_t> m_model_output_shape_tflite;
    const int m_model_latency_tflite;
    const bool m_model_stateful_tflite;
#endif
};


#endif //NN_INFERENCE_TEMPLATE_INFERENCECONFIG_H
