#ifndef NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H

#include <JuceHeader.h>

#include "InferenceThread.h"
#include "../utils/ThreadSafeBuffer.h"
#include "../utils/HostConfig.h"

class InferenceManager {
public:
    InferenceManager();
    ~InferenceManager();

    void prepareToPlay(HostConfig config);
    void processBlock(juce::AudioBuffer<float>& buffer);

    void parameterChanged(const juce::String &parameterID, float newValue);

    int getLatency() const;

    // Required for unit test
    int getNumReceivedSamples();
    bool isInitializing() const;
    InferenceThreadPool& getInferenceThread();

    int getMissingBlocks() {
        return inferenceCounter.load();;
    }

    int getSessionID() const {
        return sessionID;
    }

private:
    void processInput(juce::AudioBuffer<float>& buffer);
    void processOutput(juce::AudioBuffer<float>& buffer);

private:
    std::shared_ptr<InferenceThreadPool> inferenceThread;

    const int sessionID;
    HostConfig spec;

    bool init = true;
    int bufferCount = 0;
    int initSamples = 0;
    std::atomic<int> inferenceCounter {0};
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
