// #include "InferenceThread.h"

// InferenceThread::InferenceThread(std::counting_semaphore<1000>& s, std::unordered_map<int, std::unique_ptr<SessionElement>>& ses) : shouldExit(false), globalSemaphore(s), sessions(ses) {
//     onnxProcessor.prepareToPlay();
//     torchProcessor.prepareToPlay();
//     tfliteProcessor.prepareToPlay();
// }

// InferenceThread::~InferenceThread() {
//     stop();
// }

// void InferenceThread::start() {
//     thread = std::thread(&InferenceThread::run, this);
// }

// void InferenceThread::run() {
//     while (!shouldExit) {
//         globalSemaphore.acquire();
//         for (const auto& ses : sessions) {
//             if (ses.second->semaphore.try_acquire()) {
//                 auto& session = ses.second;
//                 if (session->currentBackend == ONNX) {
//                     onnxProcessor.process(*session);
//                 } else if (session->currentBackend == TORCH) {
//                     torchProcessor.process(*session);
//                 } else if (session->currentBackend == TFLITE) {
//                     tfliteProcessor.process(*session);
//                 }
//                 session->semaphore.release();
//             }
//         }
//         std::cout << "InferenceThread::run() acquired semaphore" << std::endl;
//     }
// }

// void InferenceThread::stop() {
//     shouldExit = true;
//     if (thread.joinable())
//     {
//         thread.join();
//     }
// }