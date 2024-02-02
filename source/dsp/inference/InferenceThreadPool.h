#ifndef NN_INFERENCE_TEMPLATE_INFERENCETHREADPOOL_H
#define NN_INFERENCE_TEMPLATE_INFERENCETHREADPOOL_H

#include <JuceHeader.h>
#include <semaphore>

#include "SessionElement.h"
#include "../utils/HostConfig.h"
#include "InferenceThread.h"
#include "InferenceConfig.h"
#include "backends/OnnxRuntimeProcessor.h"
#include "backends/LibtorchProcessor.h"
#include "backends/TFLiteProcessor.h"

class InferenceThreadPool{
public:
    InferenceThreadPool();
    ~InferenceThreadPool();
    static std::shared_ptr<InferenceThreadPool> getInstance();
    static SessionElement& createSession();
    static void releaseSession(SessionElement& session);
    static void releaseInstance();
    static void releaseThreadPool();

    static int getNumberOfSessions() {
        return activeSessions.load();
    }

    inline static std::counting_semaphore<1000> globalSemaphore{0};
    void newDataSubmitted(SessionElement& session);
    void newDataRequest(SessionElement& session, double bufferSizeInSec);

    static std::vector<std::shared_ptr<SessionElement>>& getSessions();

private:
    inline static std::shared_ptr<InferenceThreadPool> inferenceThreadPool = nullptr; 
    static int getAvailableSessionID();

    static void preProcess(SessionElement& session);
    static void postProcess(SessionElement& session, SessionElement::ThreadSafeStruct& nextBuffer);

private:
    inline static std::vector<std::shared_ptr<SessionElement>> sessions;
    inline static std::atomic<int> nextId = 0;
    inline static std::atomic<int> activeSessions = 0;
    inline static bool threadPoolShouldExit = false;

    inline static std::vector<std::unique_ptr<InferenceThread>> threadPool;
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCETHREADPOOL_H
