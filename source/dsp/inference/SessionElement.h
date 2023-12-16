//
// Created by Valentin Ackva on 16/12/2023.
//

#ifndef NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H
#define NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H

#include <JuceHeader.h>
#include "../utils/RingBuffer.h"
#include "../utils/ThreadSafeBuffer.h"
#include "InferenceConfig.h"
#include <semaphore>

struct SessionElement {
    SessionElement(int newSessionID);

    RingBuffer sendBuffer;
    RingBuffer receiveBuffer;

    struct ThreadSafeStruct {
        std::binary_semaphore free{true};
        std::binary_semaphore ready{false};
        std::binary_semaphore done{false};
        std::chrono::time_point<std::chrono::system_clock> time;
        NNInferenceTemplate::InputArray processedModelInput;
        NNInferenceTemplate::OutputArray rawModelOutputBuffer;
    };
    std::array<ThreadSafeStruct, 5000> inferenceQueue;

    std::atomic<InferenceBackend> currentBackend {ONNX};
    std::queue<std::chrono::time_point<std::chrono::system_clock>> timeStamps; // TODO remove

    std::counting_semaphore<1000> sendSemaphore{0};

    const std::atomic<int> sessionID;
};


#endif //NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H
