//
// Created by Valentin Ackva on 02/02/2024.
//

#ifndef NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H

#include "InferenceManager.h"

class InferenceHandler {
public:
    InferenceHandler() {

    }

    ~InferenceHandler() {

    }

    void setInferenceBackend(InferenceBackend inferenceBackend) {
        currentBackend = inferenceBackend;
    }

    InferenceBackend getInferenceBackend() {
        return currentBackend;
    }

    void prepare(HostAudioConfig newAudioConfig) {
        assert(newAudioConfig.hostChannels == 1 && "Stereo processing is not fully implemented yet");
        inferenceManager.prepare(newAudioConfig);
    }

    int getLatency() {
        return inferenceManager.getLatency();
    }

    // buffer[channel][index]
    void process(float ** inputBuffer, const int inputSamples) {
        inferenceManager.process(inputBuffer, inputSamples);
    }

    // TODO remove
    InferenceManager &getInferenceManager() {
        return inferenceManager;
    }

private:
    InferenceBackend currentBackend;
    InferenceManager inferenceManager;
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H
