#ifndef NN_INFERENCE_TEMPLATE_INFERENCETHREADPOOL_H
#define NN_INFERENCE_TEMPLATE_INFERENCETHREADPOOL_H

#include <JuceHeader.h>
#include <semaphore>

#include "../utils/ThreadSafeBuffer.h"
#include "../utils/HostConfig.h"
#include "InferenceThread.h"
#include "InferenceConfig.h"
#include "backends/OnnxRuntimeProcessor.h"
#include "backends/LibtorchProcessor.h"
#include "backends/TFLiteProcessor.h"

struct SessionElement {
    ThreadSafeBuffer sendBuffer {1, 48000};
    ThreadSafeBuffer receiveBuffer {1, 48000};

    NNInferenceTemplate::OutputArray rawModelOutputBuffer{};
    NNInferenceTemplate::InputArray processedModelInput{};

    std::atomic<InferenceBackend> currentBackend {ONNX};
    std::counting_semaphore<1000> semaphore{0};
};

class InferenceThreadPool{
public:
    static InferenceThreadPool& getInstance();
    static SessionElement& createSession();
    static void releaseSession(SessionElement& session);

    static int getNumberOfSessions() {
        return activeSessions.load();
    }

    static void prepareToPlay(HostConfig spec, int sessionID);
    static void setBackend(InferenceBackend backend, int sessionID);

    static ThreadSafeBuffer& getSendBuffer(int sessionID);
    static ThreadSafeBuffer& getReceiveBuffer(int sessionID);

    inline static std::counting_semaphore<1000> semaphore{0};
    void newDataSubmitted(int sessionID);

private:
    InferenceThreadPool();
    static int getAvailableSessionID();

    static void process(SessionElement& session);

    static void preProcess(SessionElement& session);
    static void inference(SessionElement& session);
    static void postProcess(SessionElement& session);

private:
    inline static std::vector<std::unique_ptr<SessionElement>> sessions;
    inline static std::atomic<int> nextId = 0;
    inline static std::atomic<int> activeSessions = 0;
    inline static bool threadPoolShouldExit = false;

    // inline static std::vector<std::unique_ptr<InferenceThread>> singleThreadPool;
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCETHREADPOOL_H
