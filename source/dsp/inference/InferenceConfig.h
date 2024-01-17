#ifndef NN_INFERENCE_TEMPLATE_INFERENCECONFIG_H
#define NN_INFERENCE_TEMPLATE_INFERENCECONFIG_H

enum InferenceBackend {
    LIBTORCH,
    ONNX,
    TFLITE
};

#if MODEL_TO_USE == 1

#define MODELS_PATH_TENSORFLOW GUITARLSTM_MODELS_PATH_TENSORFLOW
#define MODEL_TFLITE "model_0/model_0-streaming.tflite"
#define MODELS_PATH_PYTORCH GUITARLSTM_MODELS_PATH_PYTORCH
#define MODEL_LIBTORCH "model_0/model_0-streaming.pt"
#define MODELS_PATH_ONNX GUITARLSTM_MODELS_PATH_TENSORFLOW
#define MODEL_ONNX "model_0/model_0-tflite-streaming.onnx"

#define WARM_UP false
#define BATCH_SIZE 128
#define MODEL_INPUT_SIZE 1
#define MODEL_INPUT_SIZE_BACKEND 150 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_INPUT_SHAPE_ONNX {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1}
#define MODEL_INPUT_SHAPE_TFLITE {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1}
#define MODEL_INPUT_SHAPE_LIBTORCH {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND}


#define MODEL_OUTPUT_SIZE_BACKEND 1
#define MODEL_OUTPUT_SHAPE {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND}


#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 256
#endif

#define MODEL_LATENCY 0

namespace NNInferenceTemplate {
    using InputArray = std::array<float, BATCH_SIZE * MODEL_INPUT_SIZE_BACKEND>;
    using OutputArray = std::array<float, BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND>;
}

#elif MODEL_TO_USE == 2

#define MODELS_PATH_TENSORFLOW STEERABLENAFX_MODELS_PATH_TENSORFLOW
#define MODEL_TFLITE "model_0/steerable-nafx.tflite"
#define MODELS_PATH_PYTORCH STEERABLENAFX_MODELS_PATH_PYTORCH
#define MODEL_LIBTORCH "model_0/steerable-nafx.pt"
#define MODELS_PATH_ONNX STEERABLENAFX_MODELS_PATH_PYTORCH
#define MODEL_ONNX "model_0/steerable-nafx-libtorch.onnx"

#define WARM_UP false
#define BATCH_SIZE 1
#define MODEL_INPUT_SIZE 64
#define MODEL_INPUT_SIZE_BACKEND 56236 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_INPUT_SHAPE_ONNX {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND}
#define MODEL_INPUT_SHAPE_TFLITE {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1}
#define MODEL_INPUT_SHAPE_LIBTORCH {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND}


#define MODEL_OUTPUT_SIZE_BACKEND 64
#define MODEL_OUTPUT_SHAPE_TFLITE {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND, 1}
#define MODEL_OUTPUT_SHAPE_LIBTORCH {BATCH_SIZE, 1, MODEL_OUTPUT_SIZE_BACKEND}


#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 256
#endif

#define MODEL_LATENCY 0

namespace NNInferenceTemplate {
    using InputArray = std::array<float, BATCH_SIZE * MODEL_INPUT_SIZE_BACKEND>;
    using OutputArray = std::array<float, BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND>;
}

#endif // MODEL_TO_USE

#endif //NN_INFERENCE_TEMPLATE_INFERENCECONFIG_H