#include "InferenceManager.h"
#include "../PluginParameters.h"

InferenceManager::InferenceManager() : inferenceThreadPool(InferenceThreadPool::getInstance()), session(inferenceThreadPool->createSession()) {
}

InferenceManager::~InferenceManager() {
    inferenceThreadPool->releaseSession(session);
}

void InferenceManager::parameterChanged(const juce::String &parameterID, float newValue) {
    if (parameterID == PluginParameters::BACKEND_TYPE_ID.getParamID()) {
        InferenceBackend newInferenceBackend = ((int) newValue == 0) ? TFLITE :
                                               ((int) newValue == 1) ? LIBTORCH : ONNX;
        session.currentBackend = newInferenceBackend;
    }
}

void InferenceManager::prepareToPlay(HostConfig newConfig) {
    spec = newConfig;

    session.sendBuffer.initialise(1, (size_t) spec.hostSampleRate * 6);
    session.receiveBuffer.initialise(1, (size_t) spec.hostSampleRate * 6);
    inferenceCounter = 0;

    init = true;
    bufferCount = 0;

    int result = (int) spec.hostBufferSize % (BATCH_SIZE * MODEL_INPUT_SIZE);
    if (result == 0) {
        initSamples = MAX_INFERENCE_TIME + BATCH_SIZE * MODEL_LATENCY;
    } else if (result > 0 && result < (int) spec.hostBufferSize) {
        initSamples = MAX_INFERENCE_TIME + (int) spec.hostBufferSize + BATCH_SIZE * MODEL_LATENCY; //TODO not minimum possible
    } else {
        initSamples = MAX_INFERENCE_TIME + (BATCH_SIZE * MODEL_INPUT_SIZE) + BATCH_SIZE * MODEL_LATENCY;
    }
}

void InferenceManager::processBlock(juce::AudioBuffer<float> &buffer) {
    processInput(buffer);
    if (init) {
        bufferCount += buffer.getNumSamples();
        buffer.clear();
        if (bufferCount >= initSamples) init = false;
    } else {
        processOutput(buffer);
    }
}

void InferenceManager::processInput(juce::AudioBuffer<float> &buffer) {
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
        session.sendBuffer.pushSample(buffer.getSample(0, sample), 0);
    }
    inferenceThreadPool->newDataSubmitted(session);
}

void InferenceManager::processOutput(juce::AudioBuffer<float> &buffer) {
    double timeInSec = static_cast<double>(buffer.getNumSamples()) / spec.hostSampleRate;
    inferenceThreadPool->newDataRequest(session, timeInSec);
    
    while (inferenceCounter > 0) {
        if (session.receiveBuffer.getAvailableSamples(0) >= 2 * (size_t) buffer.getNumSamples()) {
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                session.receiveBuffer.popSample(0);
            }
            inferenceCounter--;
            std::cout << "##### catch up samples" << std::endl;
        }
        else {
            break;
        }
    }
    if (session.receiveBuffer.getAvailableSamples(0) >= (size_t) buffer.getNumSamples()) {
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
            buffer.setSample(0, sample, session.receiveBuffer.popSample(0));
        }
    }
    else {
        buffer.clear();
        inferenceCounter++;
        std::cout << "##### missing samples" << std::endl;
    }
}

int InferenceManager::getLatency() const {
    if (initSamples % (int) spec.hostBufferSize == 0) return initSamples;
    else return ((int) ((float) initSamples / (float) spec.hostBufferSize) + 1) * (int) spec.hostBufferSize;
}

InferenceThreadPool& InferenceManager::getInferenceThreadPool() {
    return *inferenceThreadPool;
}

int InferenceManager::getNumReceivedSamples() {
    inferenceThreadPool->newDataRequest(session, 0); // TODO: Check if processOutput call is better here
    return session.receiveBuffer.getAvailableSamples(0);
}

bool InferenceManager::isInitializing() const {
    return init;
}
