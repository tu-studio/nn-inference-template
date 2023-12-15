#include "InferenceManager.h"
#include "../PluginParameters.h"

InferenceManager::InferenceManager() : sessionID(InferenceThreadPool::getAvailableSessionID()) {
    inferenceThread = std::make_shared<InferenceThreadPool>(InferenceThreadPool::getInstance(sessionID));
}

InferenceManager::~InferenceManager() {
    inferenceThread->releaseInstance(sessionID);
}

void InferenceManager::parameterChanged(const juce::String &parameterID, float newValue) {
    if (parameterID == PluginParameters::BACKEND_TYPE_ID.getParamID()) {
        InferenceBackend newInferenceBackend = (newValue == 0.0f) ? TFLITE :
                                               (newValue == 1.f) ? LIBTORCH : ONNX;
        inferenceThread->setBackend(newInferenceBackend, sessionID);
    }
}

void InferenceManager::prepareToPlay(HostConfig newConfig) {
    spec = newConfig;

    inferenceThread->prepareToPlay(spec, sessionID);
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
    auto& sendBuffer = inferenceThread->getSendBuffer(sessionID);
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
        sendBuffer.pushSample(buffer.getSample(0, sample), 0);
    }
    inferenceThread->newDataSubmitted(sessionID);
}

void InferenceManager::processOutput(juce::AudioBuffer<float> &buffer) {
    auto& receiveBuffer = inferenceThread->getReceiveBuffer(sessionID);
    while (inferenceCounter > 0) {
        if (receiveBuffer.getAvailableSamples(0) >= 2 * buffer.getNumSamples()) {
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                receiveBuffer.popSample(0);
            }
            inferenceCounter--;
        }
        else {
            break;
        }
    }
    if (receiveBuffer.getAvailableSamples(0) >= buffer.getNumSamples()) {
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
            buffer.setSample(0, sample, receiveBuffer.popSample(0));
        }
    }
    else {
        buffer.clear();
        inferenceCounter++;
        //std::cout << "##### missing samples" << std::endl;
    }
}

int InferenceManager::getLatency() const {
    if (initSamples % (int) spec.hostBufferSize == 0) return initSamples;
    else return ((int) ((float) initSamples / (float) spec.hostBufferSize) + 1) * (int) spec.hostBufferSize;
}

InferenceThreadPool &InferenceManager::getInferenceThread() {
    return *inferenceThread;
}

int InferenceManager::getNumReceivedSamples() {
    return inferenceThread->getReceiveBuffer(sessionID).getAvailableSamples(0);
}

bool InferenceManager::isInitializing() const {
    return init;
}
