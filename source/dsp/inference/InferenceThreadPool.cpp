#include "InferenceThreadPool.h"

InferenceThreadPool::InferenceThreadPool()  {
    for (size_t i = 0; i < (size_t) std::thread::hardware_concurrency() - 1; ++i) {
        threadPool.emplace_back(std::make_unique<InferenceThread>(globalSemaphore, sessions));
    }
    std::cout << "std::thread::hardware_concurrency() - 1 " << std::thread::hardware_concurrency() - 1 << std::endl;
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
    for (size_t i = 0; i < session.inferenceQueue.size(); ++i) {
        if (! session.inferenceQueue[i].free.try_acquire()) {
            if (! session.inferenceQueue[i].ready.try_acquire()) {
                session.inferenceQueue[i].done.acquire();
            }
        }
    } 
    for (size_t i = 0; i < sessions.size(); ++i) {
        if (sessions[i].get() == &session) {
            sessions.erase(sessions.begin() + (ptrdiff_t) i);
            break;
        }
    }
    activeSessions--;
}

void InferenceThreadPool::newDataSubmitted(SessionElement& session) {
    while (session.sendBuffer.getAvailableSamples(0) >= (BATCH_SIZE * MODEL_INPUT_SIZE)) {
        preProcess(session);
        session.sendSemaphore.release();
        globalSemaphore.release();
    }
}

void InferenceThreadPool::newDataRequest(SessionElement& session, double bufferSizeInSec) {
    auto timeToProcess = std::chrono::milliseconds(static_cast<int>(bufferSizeInSec / 1000 * 0.5));
    auto currentTime = std::chrono::system_clock::now();
    auto waitUntil = currentTime + timeToProcess;

    for (size_t i = 0; i < session.inferenceQueue.size(); ++i) {
        if (session.inferenceQueue[i].time == session.timeStamps.front()) {
            if (session.inferenceQueue[i].done.try_acquire_until(waitUntil)) {
                session.timeStamps.pop();
                postProcess(session, session.inferenceQueue[i]);
            }
        }
    }
}

std::vector<std::shared_ptr<SessionElement>>& InferenceThreadPool::getSessions() {
    return sessions;
}

void InferenceThreadPool::preProcess(SessionElement& session) {
    for (size_t i = 0; i < session.inferenceQueue.size(); ++i) {
        if (session.inferenceQueue[i].free.try_acquire()) {
            // TODO if getAvSamples != 0 check
            for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
                size_t baseIdx = batch * MODEL_INPUT_SIZE_BACKEND;

                for (int j = MODEL_INPUT_SIZE_BACKEND - 1; j >= 0; j--) {
                    if (j == MODEL_INPUT_SIZE_BACKEND - 1) {
                        session.inferenceQueue[i].processedModelInput[baseIdx + (size_t) j] = session.sendBuffer.popSample(0);
                    } else  {
                        session.inferenceQueue[i].processedModelInput[baseIdx + (size_t) j] = session.sendBuffer.getSample(0, MODEL_INPUT_SIZE_BACKEND - (size_t) j);
                    }
                }
            }
            const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            session.timeStamps.push(now);
            session.inferenceQueue[i].time = now;
            session.inferenceQueue[i].ready.release();
            break;
        }
    }
}

void InferenceThreadPool::postProcess(SessionElement& session, SessionElement::ThreadSafeStruct& nextBuffer) {
    for (size_t j = 0; j < BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND; j++) {
        session.receiveBuffer.pushSample(nextBuffer.rawModelOutputBuffer[j], 0);
    }
    nextBuffer.free.release();
}
