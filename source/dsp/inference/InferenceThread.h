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
#include "SessionElement.h"

class InferenceThread {
public:
    InferenceThread(std::counting_semaphore<1000>& globalSemaphore, std::vector<std::unique_ptr<SessionElement>>& sessions);
    ~InferenceThread();

     void start();
     void run();
     void stop();

private:
    void inference(InferenceBackend backend, NNInferenceTemplate::InputArray& input, NNInferenceTemplate::OutputArray& output);

private:
     std::thread thread;
     std::atomic<bool> shouldExit;
     std::counting_semaphore<1000>& globalSemaphore;
     std::vector<std::unique_ptr<SessionElement>>& sessions;

     OnnxRuntimeProcessor onnxProcessor;
     LibtorchProcessor torchProcessor;
     TFLiteProcessor tfliteProcessor;
 };

#endif //NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H