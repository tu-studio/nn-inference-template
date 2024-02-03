#ifndef NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H
#define NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H

#include <JuceHeader.h>
#include <semaphore>

#include "../utils/HostConfig.h"
#include "InferenceConfig.h"
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
    void inference(InferenceBackend backend, NNInferenceTemplate::InputArray& input, NNInferenceTemplate::OutputArray& output);

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