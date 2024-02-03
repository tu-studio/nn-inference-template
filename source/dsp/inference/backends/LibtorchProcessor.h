#ifndef NN_INFERENCE_TEMPLATE_LIBTORCHPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_LIBTORCHPROCESSOR_H

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <torch/script.h>
#include <stdlib.h>

class LibtorchProcessor {
public:
    LibtorchProcessor();
    ~LibtorchProcessor();

    void prepareToPlay();
    void processBlock(AudioBufferF& input, AudioBufferF& output);

private:
    std::string filepath = MODELS_PATH_PYTORCH;
    std::string modelname = MODEL_LIBTORCH;

    torch::jit::script::Module module;

    at::Tensor inputTensor;
    at::Tensor outputTensor;
    std::vector<torch::jit::IValue> inputs;
};


#endif //NN_INFERENCE_TEMPLATE_LIBTORCHPROCESSOR_H
