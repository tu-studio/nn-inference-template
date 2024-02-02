#ifndef NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H

#include <JuceHeader.h>

#include "InferenceThread.h"
#include "InferenceThreadPool.h"
#include "../utils/HostConfig.h"

class InferenceManager {
public:
    InferenceManager();
    ~InferenceManager();

    void prepare(HostAudioConfig config);
    void process(float ** inputBuffer, int inputSamples);

    void parameterChanged(const juce::String &parameterID, float newValue);

    int getLatency() const;

    // Required for unit test
    int getNumReceivedSamples();
    bool isInitializing() const;
    InferenceThreadPool& getInferenceThreadPool();

    int getMissingBlocks();
    int getSessionID() const;

private:
    void processInput(float ** inputBuffer, const int inputSamples);
    void processOutput(float ** inputBuffer, const int inputSamples);
    void clearBuffer(float ** inputBuffer, const int inputSamples);

private:
    std::shared_ptr<InferenceThreadPool> inferenceThreadPool;

    SessionElement& session;
    HostAudioConfig spec;

    bool init = true;
    int bufferCount = 0;
    int initSamples = 0;
    std::atomic<int> inferenceCounter {0};
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEMANAGER_H
