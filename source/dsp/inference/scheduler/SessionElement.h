#ifndef NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H
#define NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H

#include <semaphore>
#include <queue>
#include <atomic>
#include <InferenceConfig.h>
#include <RingBuffer.h>
#include <InferenceBackend.h>
#include <PrePostProcessor.h>

struct SessionElement {
    SessionElement(int newSessionID, PrePostProcessor& prePostProcessor);

    RingBuffer sendBuffer;
    RingBuffer receiveBuffer;

    struct ThreadSafeStruct {
        std::binary_semaphore free{true};
        std::binary_semaphore ready{false};
        std::binary_semaphore done{false};
        std::chrono::time_point<std::chrono::system_clock> time;
        NNInferenceTemplate::InputArray processedModelInput;
        NNInferenceTemplate::OutputArray rawModelOutput;
    };
    std::array<ThreadSafeStruct, 5000> inferenceQueue;

    std::atomic<InferenceBackend> currentBackend {ONNX};
    std::queue<std::chrono::time_point<std::chrono::system_clock>> timeStamps;
    std::counting_semaphore<1000> sendSemaphore{0};
    
    const int sessionID;

    PrePostProcessor& prePostProcessor;
};


#endif //NN_INFERENCE_TEMPLATE_SESSIONELEMENT_H
