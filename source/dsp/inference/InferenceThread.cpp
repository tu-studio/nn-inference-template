#include "InferenceThread.h"

InferenceThread::InferenceThread(std::counting_semaphore<1000>& s, std::vector<std::unique_ptr<SessionElement>>& ses) : shouldExit(false), globalSemaphore(s), sessions(ses) {
    onnxProcessor.prepareToPlay();
    torchProcessor.prepareToPlay();
    tfliteProcessor.prepareToPlay();
    start();
}

InferenceThread::~InferenceThread() {
    stop();
}

void InferenceThread::start() {
    thread = std::thread(&InferenceThread::run, this);
}

void InferenceThread::run() {
    while (!shouldExit) {
        globalSemaphore.acquire();
        std::cout << "InferenceThread::run() acquired semaphore" << std::endl;
        for (const auto& ses : sessions) {
            if (ses->sendSemaphore.try_acquire()) {
                if (ses->currentBackend == ONNX) {
                    // onnxProcessor.processBlock(ses->processedModelInput, ses->rawModelOutputBuffer);
                    std::cout << "processing data darling" << std::endl;
                } else if (ses->currentBackend == LIBTORCH) {
                    torchProcessor.processBlock(ses->processedModelInput, ses->rawModelOutputBuffer);
                } else if (ses->currentBackend == TFLITE) {
                    tfliteProcessor.processBlock(ses->processedModelInput, ses->rawModelOutputBuffer);
                }
            }
        }
    }
}

void InferenceThread::stop() {
    shouldExit = true;
    if (thread.joinable())
    {
        thread.join();
    }
}