#include "InferenceThread.h"

InferenceThread::InferenceThread(std::counting_semaphore<1000>& s, std::vector<std::shared_ptr<SessionElement>>& ses) : shouldExit(false), globalSemaphore(s), sessions(ses) {
    onnxProcessor.prepareToPlay();
    torchProcessor.prepareToPlay();
    tfliteProcessor.prepareToPlay();
    std::cout << "starting thread" << std::endl;
    start();
}

InferenceThread::~InferenceThread() {
    std::cout << "stopping thread" << std::endl;
    stop();
}

void InferenceThread::start() {
    thread = std::thread(&InferenceThread::run, this);
}

void InferenceThread::run() {
    std::chrono::milliseconds timeForExit(100);
    while (!shouldExit) {
        auto success = globalSemaphore.try_acquire_for(timeForExit);
        for (const auto& session : sessions) {
            if (session->sendSemaphore.try_acquire()) {
                for (size_t i = 0; i < session->inferenceQueue.size(); ++i) {
                    if (session->inferenceQueue[i].ready.try_acquire()) {
                        inference(session->currentBackend, session->inferenceQueue[i].processedModelInput, session->inferenceQueue[i].rawModelOutputBuffer);
                        session->inferenceQueue[i].done.release();
                        break;
                    }
                }
                break;
            }
        }
    }
}

void InferenceThread::inference(InferenceBackend backend, NNInferenceTemplate::InputArray &input, NNInferenceTemplate::OutputArray &output) {
    if (backend == ONNX) {
        onnxProcessor.processBlock(input, output);
    } else if (backend == LIBTORCH) {
        torchProcessor.processBlock(input, output);
    } else if (backend == TFLITE) {
        tfliteProcessor.processBlock(input, output);
    }
}


void InferenceThread::stop() {
    shouldExit = true;
    thread.join();
}
