//
// Created by Valentin Ackva on 02/02/2024.
//

#include "InferenceHandler.h"

InferenceHandler::InferenceHandler() {

}

InferenceHandler::~InferenceHandler() {

}

void InferenceHandler::prepare(HostAudioConfig newAudioConfig) {
    assert(newAudioConfig.hostChannels == 1 && "Stereo processing is not fully implemented yet");
    inferenceManager.prepare(newAudioConfig);
}

void InferenceHandler::process(float **inputBuffer, const size_t inputSamples) {
    inferenceManager.process(inputBuffer, inputSamples);
}

void InferenceHandler::setInferenceBackend(InferenceBackend inferenceBackend) {
    inferenceManager.setBackend(inferenceBackend);
}

InferenceBackend InferenceHandler::getInferenceBackend() {
    return inferenceManager.getBackend();
}

int InferenceHandler::getLatency() {
    return inferenceManager.getLatency();
}

InferenceManager &InferenceHandler::getInferenceManager() {
    return inferenceManager;
}


