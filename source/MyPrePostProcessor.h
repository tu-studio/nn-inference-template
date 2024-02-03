#ifndef NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H

#include <PrePostProcessor.h>
#include "Configs.h"
class MyPrePostProcessor : public PrePostProcessor
{
public:
#if MODEL_TO_USE == 1
    virtual void preProcess(RingBuffer& input, NNInferenceTemplate::InputArray& output, InferenceBackend currentInferenceBackend) override {
        for (size_t batch = 0; batch < config.m_batch_size; batch++) {
            size_t baseIdx = batch * config.m_model_input_size_backend;
            popSamplesFromBuffer(input, output, config.m_model_input_size, config.m_model_input_size_backend-config.m_model_input_size, baseIdx);
        }
        std::ignore = currentInferenceBackend;
    };
#elif MODEL_TO_USE == 2
    virtual void preProcess(RingBuffer& input, NNInferenceTemplate::InputArray& output, InferenceBackend currentInferenceBackend) override {
        popSamplesFromBuffer(input, output, config.m_model_input_size, config.m_model_input_size_backend-config.m_model_input_size);
        std::ignore = currentInferenceBackend;
    };
#elif MODEL_TO_USE == 3
    // The third model uses the default preProcess method
#endif // MODEL_TO_USE
};

#endif // NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H