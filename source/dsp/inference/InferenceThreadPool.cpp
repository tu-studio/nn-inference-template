#include "InferenceThreadPool.h"

InferenceThreadPool::InferenceThreadPool()  {
    for (size_t i = 0; i < (size_t) std::thread::hardware_concurrency(); ++i) {
        threadPool.emplace_back(std::make_unique<InferenceThread>(globalSemaphore, sessions));
    }
}

int InferenceThreadPool::getAvailableSessionID() {
    nextId++;
    activeSessions++;
    return nextId.load();
}

InferenceThreadPool& InferenceThreadPool::getInstance() {
    static InferenceThreadPool instance;
    return instance;
}

SessionElement& InferenceThreadPool::createSession() {
    int sessionID = getAvailableSessionID();
    sessions.emplace_back(std::make_unique<SessionElement>(sessionID));

    return *sessions.back();
}

void InferenceThreadPool::releaseSession(SessionElement& session) {
    activeSessions--;
    for (int i = 0; i < sessions.size(); ++i) {
        if (sessions[i].get() == &session) {
            sessions.erase(sessions.begin() + i);
            break;
        }
    }   
}

void InferenceThreadPool::newDataSubmitted(SessionElement& session) {
    while (session.sendBuffer.getAvailableSamples(0) >= (BATCH_SIZE * MODEL_INPUT_SIZE)) {
        preProcess(session);
        session.sendSemaphore.release();
        globalSemaphore.release();
    }
}

void InferenceThreadPool::process(SessionElement& session) {
    preProcess(session);
    inference(session);
    postProcess(session);
}

void InferenceThreadPool::preProcess(SessionElement& session) {
    for (int i = 0; i < session.inferenceQueue.size(); ++i) {
        if (session.inferenceQueue[i].free.try_acquire()) {
            // TODO if getAvSamples != 0 check
            for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
                size_t baseIdx = batch * MODEL_INPUT_SIZE_BACKEND;
                size_t prevBaseIdx = (batch == 0 ? BATCH_SIZE - 1 : batch - 1) * MODEL_INPUT_SIZE_BACKEND;

                for (size_t j = 1; j < MODEL_INPUT_SIZE_BACKEND; j++) {
                    session.inferenceQueue[i].processedModelInput = session.inferenceQueue[i].processedModelInput[prevBaseIdx + j];
                }

                session.inferenceQueue[i].processedModelInput[baseIdx + MODEL_INPUT_SIZE_BACKEND - 1] = session.sendBuffer.popSample(0);
            }
            session.inferenceQueue[i].time = std::chrono::system_clock::now();
            break;
        }
    }
}

void InferenceThreadPool::inference(SessionElement& session) {
//    pool.submit([&session] {
//        session.processed = true;
//    });


    for (int i = 0; i < session.processedModelInput.size(); ++i) {
        session.rawModelOutputBuffer[i] = session.processedModelInput[i];
    }

/*    if (session.currentBackend == ONNX) {
        onnxProcessor.processBlock(session.processedModelInput, session.rawModelOutputBuffer);
    } else if (session.currentBackend == LIBTORCH) {
        torchProcessor.processBlock(session.processedModelInput, session.rawModelOutputBuffer);
    } else if (session.currentBackend == TFLITE) {
        tfliteProcessor.processBlock(session.processedModelInput, session.rawModelOutputBuffer);
    }*/
}

void InferenceThreadPool::postProcess(SessionElement& session) {
    for (size_t j = 0; j < BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND; j++) {
        session.receiveBuffer.pushSample(session.rawModelOutputBuffer[j], 0);
    }
    //session.processing = false;
}
