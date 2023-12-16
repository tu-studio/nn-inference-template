// #ifndef NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H
// #define NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H

// #include <JuceHeader.h>
// #include <semaphore>

// #include "../utils/ThreadSafeBuffer.h"
// #include "../utils/HostConfig.h"
// #include "InferenceConfig.h"
// #include "backends/OnnxRuntimeProcessor.h"
// #include "backends/LibtorchProcessor.h"
// #include "backends/TFLiteProcessor.h"

// class InferenceThread {
// public:
//     InferenceThread(std::counting_semaphore<1000>& globalSemaphore, std::unordered_map<int, std::unique_ptr<SessionElement>>& sessions);
//     ~InferenceThread();

//     void start();
//     void run();
//     void stop();

// private:
//     std::thread thread;
//     std::atomic<bool> shouldExit;
//     std::counting_semaphore<1000>& globalSemaphore;
//     std::unordered_map<int, std::unique_ptr<SessionElement>>& sessions;

//     OnnxRuntimeProcessor onnxProcessor;
//     LibtorchProcessor torchProcessor;
//     TFLiteProcessor tfliteProcessor;
// };

// #endif //NN_INFERENCE_TEMPLATE_INFERENCETHREAD_H