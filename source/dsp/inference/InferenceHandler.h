//
// Created by Valentin Ackva on 02/02/2024.
//

#ifndef NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H

#include "InferenceManager.h"

class InferenceHandler {
public:
    InferenceHandler();
    ~InferenceHandler();

    void setInferenceBackend(InferenceBackend inferenceBackend);
    InferenceBackend getInferenceBackend();

    void prepare(HostAudioConfig newAudioConfig);
    void process(float ** inputBuffer, const size_t inputSamples); // buffer[channel][index]

    int getLatency();
    InferenceManager &getInferenceManager(); // TODO remove

private:
    InferenceBackend currentBackend;
    InferenceManager inferenceManager;
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H
