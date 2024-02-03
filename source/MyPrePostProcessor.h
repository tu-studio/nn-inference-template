#ifndef NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H

#include <PrePostProcessor.h>

class MyPrePostProcessor : public PrePostProcessor
{
public:
#if MODEL_TO_USE == 1
    virtual void preProcess(RingBuffer& input, NNInferenceTemplate::InputArray& output, [[maybe_unused]] InferenceBackend currentInferenceBackend) override {
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            size_t baseIdx = batch * MODEL_INPUT_SIZE_BACKEND;
            popSamplesFromBuffer(input, output, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE_BACKEND-MODEL_INPUT_SIZE, baseIdx);
        }
    };
#elif MODEL_TO_USE == 2
    virtual void preProcess(RingBuffer& input, NNInferenceTemplate::InputArray& output, [[maybe_unused]] InferenceBackend currentInferenceBackend) override {
        popSamplesFromBuffer(input, output, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE_BACKEND-MODEL_INPUT_SIZE);
    };
#elif MODEL_TO_USE == 3
    // The third model uses the default preProcess method
#endif // MODEL_TO_USE
};

#endif // NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H