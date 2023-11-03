#ifndef NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H

#include <JuceHeader.h>

#include "InferenceThread.h"
#include "../utils/RingBuffer.h"

class InferenceManager {
public:
    InferenceManager();
    ~InferenceManager();

    void prepareToPlay(const juce::dsp::ProcessSpec& spec);
    void processBlock(juce::AudioBuffer<float>& buffer);

    void parameterChanged(const juce::String &parameterID, float newValue);

    InferenceThread &getInferenceThread();
    int getLatency();

    int getNumReceivedSamples() {
        return receiveRingBuffer.getAvailableSamples(0);
    }

private:
    void processInput(juce::AudioBuffer<float>& buffer);
    void processOutput(juce::AudioBuffer<float>& buffer);

private:
    bool init = true;
    int bufferCount = 0;
    int initSamples = 0;

    InferenceThread inferenceThread;

    RingBuffer sendRingBuffer;
    RingBuffer receiveRingBuffer;

    std::atomic<int> numInferencedBufferAvailable;

    juce::dsp::ProcessSpec spec;

    int inferenceCounter = 0;
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
