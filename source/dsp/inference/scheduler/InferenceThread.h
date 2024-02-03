#ifndef NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H
#define NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H

#include <JuceHeader.h>
#include <semaphore>

#include "InferenceConfig.h"
#include "backends/OnnxRuntimeProcessor.h"
#include "backends/LibtorchProcessor.h"
#include "backends/TFLiteProcessor.h"
#include "SessionElement.h"
#include "../utils/AudioBuffer.h"

#if WIN32
    #include <windows.h>
#else
    #include <pthread.h>
#endif

class InferenceThread {
public:
    InferenceThread(std::counting_semaphore<1000>& globalSemaphore, std::vector<std::shared_ptr<SessionElement>>& sessions);
    ~InferenceThread();

    void start();
    void run();
    void stop();

    void setRealTimeOrLowerPriority();

private:
    void inference(InferenceBackend backend, AudioBufferF& input, AudioBufferF& output);

private:
     std::thread thread;
     std::atomic<bool> shouldExit;
     std::counting_semaphore<1000>& globalSemaphore;
     std::vector<std::shared_ptr<SessionElement>>& sessions;

     OnnxRuntimeProcessor onnxProcessor;
     LibtorchProcessor torchProcessor;
     TFLiteProcessor tfliteProcessor;
 };

#endif //NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H