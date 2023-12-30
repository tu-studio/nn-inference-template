#include "InferenceThread.h"

InferenceThread::InferenceThread(std::counting_semaphore<1000>& s, std::vector<std::shared_ptr<SessionElement>>& ses) : shouldExit(false), globalSemaphore(s), sessions(ses) {
    onnxProcessor.prepareToPlay();
    torchProcessor.prepareToPlay();
    tfliteProcessor.prepareToPlay();
    std::cout << "starting thread" << std::endl;
}

InferenceThread::~InferenceThread() {
    std::cout << "stopping thread" << std::endl;
    stop();
}

void InferenceThread::start() {
    shouldExit = false;
    thread = std::thread(&InferenceThread::run, this);
    setRealTimeOrLowerPriority();
}

void InferenceThread::run() {
    std::chrono::milliseconds timeForExit(1);
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
    if (thread.joinable()) thread.join();
}

void InferenceThread::setRealTimeOrLowerPriority() {
#if WIN32
    int priorities[] = {THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_ABOVE_NORMAL};

    for (int priority : priorities) {
        if (SetThreadPriority(thread.native_handle(), priority)) {
            std::cout << "Thread priority set to " << priority << std::endl;
            return;
        } else {
            std::cerr << "Failed to set thread priority " << priority << std::endl;
        }
    }
#else
    //TODO test code
    sched_param sch_params;
            sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
            if(pthread_setschedparam(thread.native_handle(), SCHED_FIFO, &sch_params)) {
                std::cerr << "Failed to set Thread scheduling : " << errno << std::endl;
            }
#endif
}
