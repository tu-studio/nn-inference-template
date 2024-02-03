#ifndef NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H
#define NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H

#include <semaphore>
#include "InferenceBuffer.h"
#ifdef USE_LIBTORCH
    #include "backends/LibtorchProcessor.h"
#endif
#ifdef USE_ONNXRUNTIME
    #include "backends/OnnxRuntimeProcessor.h"
#endif
#ifdef USE_TFLITE
    #include "backends/TFLiteProcessor.h"
#endif

#include "SessionElement.h"
#include "../utils/AudioBuffer.h"

#if WIN32
    #include <windows.h>
#else
    #include <pthread.h>
#endif

class InferenceThread {
public:
    InferenceThread(std::counting_semaphore<1000>& globalSemaphore, std::vector<std::shared_ptr<SessionElement>>& sessions, InferenceConfig& config);
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

#ifdef USE_LIBTORCH
    LibtorchProcessor torchProcessor;
#endif
#ifdef USE_ONNXRUNTIME
    OnnxRuntimeProcessor onnxProcessor;
#endif
#ifdef USE_TFLITE
    TFLiteProcessor tfliteProcessor;
#endif

 };

#endif //NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H