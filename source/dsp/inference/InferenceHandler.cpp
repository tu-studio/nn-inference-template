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

void InferenceHandler::process(float **inputBuffer, const int inputSamples) {
    inferenceManager.process(inputBuffer, inputSamples);
}

void InferenceHandler::setInferenceBackend(InferenceBackend inferenceBackend) {
    currentBackend = inferenceBackend;
}

InferenceBackend InferenceHandler::getInferenceBackend() {
    return currentBackend;
}

int InferenceHandler::getLatency() {
    return inferenceManager.getLatency();
}

InferenceManager &InferenceHandler::getInferenceManager() {
    return inferenceManager;
}


