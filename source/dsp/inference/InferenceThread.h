#ifndef NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H
#define NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H

#include <JuceHeader.h>
#include <semaphore>

#include "../utils/ThreadSafeBuffer.h"
#include "../utils/HostConfig.h"
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

    std::atomic<float> processingTime;

    std::atomic<bool> processing = false;
    bool processed = false;

    OnnxRuntimeProcessor onnxProcessor;
    LibtorchProcessor torchProcessor;
    TFLiteProcessor tfliteProcessor;
};

class MyThread
{
public:
    MyThread() : shouldExit(false)
    {
    }

    void start()
    {
        thread = std::thread(&MyThread::run, this);
    }

    void run()
    {
        while (!shouldExit)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void stop()
    {
        shouldExit = true;
        if (thread.joinable())
        {
            thread.join();
        }
    }

private:
    std::thread thread;
    std::atomic<bool> shouldExit;
};

class InferenceThreadPool{
public:
    static int getAvailableSessionID();
    static InferenceThreadPool& getInstance(int sessionID) {
        static InferenceThreadPool instance;
        sessions.insert({sessionID, std::make_unique<SessionElement>()});

        sessions.at(sessionID)->onnxProcessor.prepareToPlay();
        sessions.at(sessionID)->torchProcessor.prepareToPlay();
        sessions.at(sessionID)->tfliteProcessor.prepareToPlay();

        return instance;
    }
    static void releaseInstance(int sessionID) {
        activeSessions--;
        sessions.erase(sessionID);
    }

    static int getNumberOfSessions() {
        return activeSessions.load();
    }

    static void prepareToPlay(HostConfig spec, int sessionID);
    static void setBackend(InferenceBackend backend, int sessionID);

    static ThreadSafeBuffer& getSendBuffer(int sessionID);
    static ThreadSafeBuffer& getReceiveBuffer(int sessionID);

    void newDataSubmitted(int sessionID);

private:
    InferenceThreadPool();

/*
    void run() override;
*/


    static void process(SessionElement& session);

    static void preProcess(SessionElement& session);
    static void inference(SessionElement& session);
    static void postProcess(SessionElement& session);

private:
    inline static std::unordered_map<int, std::unique_ptr<SessionElement>> sessions;
    inline static std::atomic<int> nextId = 0;
    inline static std::atomic<int> activeSessions = 0;
    inline static bool threadPoolShouldExit = false;


    inline static std::vector<std::unique_ptr<MyThread>> singleThreadPool;
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H
