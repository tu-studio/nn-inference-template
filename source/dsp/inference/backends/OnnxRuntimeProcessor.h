#ifndef NN_INFERENCE_TEMPLATE_ONNXRUNTIMEPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_ONNXRUNTIMEPROCESSOR_H

#include <JuceHeader.h>
#include "../InferenceBuffer.h"
#include <InferenceConfig.h>
#include "onnxruntime_cxx_api.h"

class OnnxRuntimeProcessor {
public:
    OnnxRuntimeProcessor(InferenceConfig& config);
    ~OnnxRuntimeProcessor();

    void prepareToPlay();
    void processBlock(NNInferenceTemplate::InputArray& input, NNInferenceTemplate::OutputArray& output);

private:
    InferenceConfig& inferenceConfig;

    Ort::Env env;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    std::vector<int64_t> inputShape;
    std::array<const char *, 1> inputNames;

    std::array<const char *, 1> outputNames;
    // Define output tensor vector
    std::vector<Ort::Value> outputTensors;
};

#endif //NN_INFERENCE_TEMPLATE_ONNXRUNTIMEPROCESSOR_H
