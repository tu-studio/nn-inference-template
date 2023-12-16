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
    std::cout << "SendBuffer available samples: " << session.sendBuffer.getAvailableSamples(0) << std::endl;
    while (session.sendBuffer.getAvailableSamples(0) >= (BATCH_SIZE * MODEL_INPUT_SIZE)) {
        preProcess(session);
        session.sendSemaphore.release();
        globalSemaphore.release();
    }
}

void InferenceThreadPool::newDataRequest(SessionElement& session) {
    if (session.returnSemaphore.try_acquire_for(std::chrono::milliseconds(1))) {
        postProcess(session);
    }
}

void InferenceThreadPool::preProcess(SessionElement& session) {
    for (int i = 0; i < session.inferenceQueue.size(); ++i) {
        if (session.inferenceQueue[i].free.try_acquire()) {
            // TODO if getAvSamples != 0 check
            for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
                size_t baseIdx = batch * MODEL_INPUT_SIZE_BACKEND;
                size_t prevBaseIdx = (batch == 0 ? BATCH_SIZE - 1 : batch - 1) * MODEL_INPUT_SIZE_BACKEND;

                for (int j = MODEL_INPUT_SIZE_BACKEND - 1; j >= 0; j--) {
                    if (j == MODEL_INPUT_SIZE_BACKEND - 1) {
                        session.inferenceQueue[i].processedModelInput[baseIdx + j] = session.sendBuffer.popSample(0);
                    } else  {
                        session.inferenceQueue[i].processedModelInput[baseIdx + j] = session.sendBuffer.getSample(0, MODEL_INPUT_SIZE_BACKEND - j);
                    }
                }
            }

            session.inferenceQueue[i].time = std::chrono::system_clock::now();
            session.inferenceQueue[i].ready.release();
            break;
        }
    }
}

void InferenceThreadPool::postProcess(SessionElement& session) {
    for (int i = 0; i < session.inferenceQueue.size(); ++i) {
        if (session.inferenceQueue[i].done.try_acquire()) {
            for (size_t j = 0; j < BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND; j++) {
                session.receiveBuffer.pushSample(session.inferenceQueue[i].rawModelOutputBuffer[j], 0);
            }
            session.inferenceQueue[i].free.release();
            break;
        }
    }
}