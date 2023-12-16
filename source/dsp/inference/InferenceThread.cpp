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
        for (const auto& session : sessions) {
            if (session->sendSemaphore.try_acquire()) {
                for (int i = 0; i < session->inferenceQueue.size(); ++i) {
                    if (session->inferenceQueue[i].ready.try_acquire()) {
                        inference(session->currentBackend, session->inferenceQueue[i].processedModelInput, session->inferenceQueue[i].rawModelOutputBuffer);
                        session->inferenceQueue[i].done.release();
                        session->returnSemaphore.release();
                        break;
                    }
                }
            }
        }
    }
}

void InferenceThread::inference(InferenceBackend backend, NNInferenceTemplate::InputArray &input, NNInferenceTemplate::OutputArray &output) {
    if (backend == ONNX) {
        // onnxProcessor.processBlock(input, output);
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            size_t baseIdx = batch * MODEL_INPUT_SIZE_BACKEND;
            output[batch] = input[baseIdx + MODEL_INPUT_SIZE_BACKEND - 1 ];
        }
    } else if (backend == LIBTORCH) {
        torchProcessor.processBlock(input, output);
    } else if (backend == TFLITE) {
        tfliteProcessor.processBlock(input, output);
    }
}


void InferenceThread::stop() {
    shouldExit = true;
    if (thread.joinable())
    {
        thread.join();
    }
}

