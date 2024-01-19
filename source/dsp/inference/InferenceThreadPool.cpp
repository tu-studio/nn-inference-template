#include "InferenceThreadPool.h"

InferenceThreadPool::InferenceThreadPool()  {
    for (size_t i = 0; i < (size_t) std::thread::hardware_concurrency() - 1; ++i) {
        threadPool.emplace_back(std::make_unique<InferenceThread>(globalSemaphore, sessions));
    }
}

InferenceThreadPool::~InferenceThreadPool() {}

int InferenceThreadPool::getAvailableSessionID() {
    nextId++;
    activeSessions++;
    return nextId.load();
}

std::shared_ptr<InferenceThreadPool> InferenceThreadPool::getInstance() {
    if (inferenceThreadPool == nullptr) {
        inferenceThreadPool = std::make_shared<InferenceThreadPool>();
    }
    return inferenceThreadPool;
}

void InferenceThreadPool::releaseInstance() {
    inferenceThreadPool.reset();
}

SessionElement& InferenceThreadPool::createSession() {
    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->stop();
    }

    int sessionID = getAvailableSessionID();
    sessions.emplace_back(std::make_unique<SessionElement>(sessionID));

    for (size_t i = 0; i < (size_t) threadPool.size(); ++i) {
        threadPool[i]->start();
    } 

    return *sessions.back();
}

void InferenceThreadPool::releaseThreadPool() {
    threadPool.clear();
}

void InferenceThreadPool::releaseSession(SessionElement& session) {
    activeSessions--;
    if (activeSessions == 0) {
        releaseThreadPool();
    }
    for (size_t i = 0; i < sessions.size(); ++i) {
        if (sessions[i].get() == &session) {
            sessions.erase(sessions.begin() + (ptrdiff_t) i);
            break;
        }
    }
    if (activeSessions == 0) {
        releaseInstance();
    }
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
        // TODO: find better way to do this fix of SEGFAULT when comparing with empty TimeStampQueue
        if (session.timeStamps.size() > 0 && session.inferenceQueue[i].time == session.timeStamps.front()) {
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
#if MODEL_TO_USE == 1
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
#elif MODEL_TO_USE == 2
            for (int j = MODEL_INPUT_SIZE_BACKEND - 1; j >= 0; j--) {
                if (j >= MODEL_INPUT_SIZE_BACKEND - MODEL_INPUT_SIZE) {
                    session.inferenceQueue[i].processedModelInput[(size_t) 2 * MODEL_INPUT_SIZE_BACKEND - j - MODEL_INPUT_SIZE - 1] = session.sendBuffer.popSample(0); // looks crazy, but this way the samples poped first are at the beginning of the end of the array
                } else  {
                    session.inferenceQueue[i].processedModelInput[(size_t) j] = session.sendBuffer.getSample(0, MODEL_INPUT_SIZE_BACKEND - (size_t) j);
                }
            }
#endif // MODEL_TO_USE

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
