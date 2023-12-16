//
// Created by Valentin Ackva on 16/12/2023.
//

#ifndef NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H
#define NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H

#include "../utils/ThreadSafeBuffer.h"
#include "InferenceConfig.h"
#include <semaphore>

struct SessionElement {
    SessionElement(int newSessionID);

    ThreadSafeBuffer sendBuffer {1, 48000};
    ThreadSafeBuffer receiveBuffer {1, 48000};

    NNInferenceTemplate::OutputArray rawModelOutputBuffer{};
    NNInferenceTemplate::InputArray processedModelInput{};

    std::atomic<InferenceBackend> currentBackend {ONNX};

    std::counting_semaphore<1000> sendSemaphore{0};
    std::counting_semaphore<1000> returnSemaphore{0};

    const std::atomic<int> sessionID;
};


#endif //NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H
