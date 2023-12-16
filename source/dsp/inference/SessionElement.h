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

    struct ThreadSafeStruct {
        std::binary_semaphore free{true};
        std::chrono::time_point<std::chrono::system_clock> time;
        ThreadSafeBuffer processedModelInput {1, BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND};
        ThreadSafeBuffer rawModelOutputBuffer {1, BATCH_SIZE * MODEL_INPUT_SIZE_BACKEND};
    };
    std::array<ThreadSafeStruct, 5000> inferenceQueue;


    std::atomic<InferenceBackend> currentBackend {ONNX};

    std::counting_semaphore<1000> sendSemaphore{0};
    std::counting_semaphore<1000> returnSemaphore{0};

    const std::atomic<int> sessionID;
};


#endif //NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H
