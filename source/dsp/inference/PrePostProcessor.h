#ifndef NN_INFERENCE_TEMPLATE_PREPOSTPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_PREPOSTPROCESSOR_H

#include <InferenceConfig.h>
#include <RingBuffer.h>
#include <InferenceBackend.h>

class PrePostProcessor
{
public:
    PrePostProcessor() = default;
    ~PrePostProcessor() = default;

    virtual void preProcess(RingBuffer& input, NNInferenceTemplate::InputArray& output, [[maybe_unused]] InferenceBackend currentInferenceBackend) {
        popSamplesFromBuffer(input, output);
    };
    virtual void postProcess(NNInferenceTemplate::OutputArray input, RingBuffer& output, [[maybe_unused]] InferenceBackend currentInferenceBackend) {
        pushSamplesToBuffer(input, output);
    }

protected:
    void popSamplesFromBuffer(RingBuffer& input, NNInferenceTemplate::InputArray& output) {
        for (size_t j = 0; j < output.size(); j++) {
            output[j] = input.popSample(0);
        }
    }

    void popSamplesFromBuffer(RingBuffer& input, NNInferenceTemplate::InputArray& output, int numNewSamples, int numOldSamples) {
        popSamplesFromBuffer(input, output, numNewSamples, numOldSamples, 0);
    }

    void popSamplesFromBuffer(RingBuffer& input, NNInferenceTemplate::InputArray& output, int numNewSamples, int numOldSamples, int offset) {
        int numTotalSamples = numNewSamples + numOldSamples;
        for (int j = numTotalSamples - 1; j >= 0; j--) {
            if (j >= numOldSamples) {
                output[(size_t) (numTotalSamples - j + numOldSamples - 1 + offset)] = input.popSample(0);
            } else  {
                output[(size_t) (j + offset)] = input.getSampleFromTail(0, (size_t) (numTotalSamples - j));
            }
        }
    }

    void pushSamplesToBuffer(const NNInferenceTemplate::OutputArray& input, RingBuffer& output) {
        for (size_t j = 0; j < input.size(); j++) {
            output.pushSample(input[j], 0);
        }
    }
};

#endif // NN_INFERENCE_TEMPLATE_PREPOSTPROCESSOR_H