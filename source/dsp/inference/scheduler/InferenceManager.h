#ifndef NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H

#include <JuceHeader.h>

#include "InferenceThread.h"
#include "InferenceThreadPool.h"
#include "../utils/HostAudioConfig.h"

class InferenceManager {
public:
    InferenceManager();
    ~InferenceManager();

    void prepare(HostAudioConfig config);
    void process(float ** inputBuffer, size_t inputSamples);

    void parameterChanged(const juce::String &parameterID, float newValue);

    int getLatency() const;

    // Required for unit test
    size_t getNumReceivedSamples();
    bool isInitializing() const;
    InferenceThreadPool& getInferenceThreadPool();

    int getMissingBlocks();
    int getSessionID() const;

private:
    void processInput(float ** inputBuffer, const size_t inputSamples);
    void processOutput(float ** inputBuffer, const size_t inputSamples);
    void clearBuffer(float ** inputBuffer, const size_t inputSamples);

private:
    std::shared_ptr<InferenceThreadPool> inferenceThreadPool;

    SessionElement& session;
    HostAudioConfig spec;

    bool init = true;
    size_t bufferCount = 0;
    size_t initSamples = 0;
    std::atomic<int> inferenceCounter {0};
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
