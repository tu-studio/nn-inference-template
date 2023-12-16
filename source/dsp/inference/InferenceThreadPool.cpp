#include "InferenceThreadPool.h"

InferenceThreadPool::InferenceThreadPool()  {
    for (size_t i = 0; i < (size_t) std::thread::hardware_concurrency(); ++i) {
         singleThreadPool.push_back(std::make_unique<InferenceThread>(semaphore, sessions));
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

int InferenceThreadPool::createSession() {
    int sessionID = getAvailableSessionID();
    sessions.insert({sessionID, std::make_unique<SessionElement>(sessionID)});

    return sessionID;
}

void InferenceThreadPool::releaseSession(int sessionID) {
    activeSessions--;
    sessions.erase(sessionID);
}

void InferenceThreadPool::prepareToPlay(HostConfig config, int sessionID) {
    auto session = sessions.at(sessionID).get();

    session->sendBuffer.initialise(1, (int) config.hostSampleRate * 6);
    session->receiveBuffer.initialise(1, (int) config.hostSampleRate * 6);
}

void InferenceThreadPool::newDataSubmitted(int sessionID) {
    // auto session = sessions.at(sessionID).get();

    // if (session->sendBuffer.getAvailableSamples(0) >= (BATCH_SIZE * MODEL_INPUT_SIZE)) {
    //     session->semaphore.release();
    //     semaphore.release();
    // }
}

/*void InferenceThreadPool::run() {
    while (!threadShouldExit()) {
        bool processIdle = true;
        for (const auto& ses : sessions) {
            auto& session = ses.second;

//            if (session->processed) {
//                session->processed = false;
//                postProcess(*session);
//            }

            if (session->sendBuffer.getAvailableSamples(0) >= (BATCH_SIZE * MODEL_INPUT_SIZE)) {
                processIdle = false;
//                session->processing = true;
//                process(*session);
                process(*session);
            }
        }
        if (processIdle) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}*/

void InferenceThreadPool::process(SessionElement& session) {
    preProcess(session);
    inference(session);
    postProcess(session);
}

void InferenceThreadPool::preProcess(SessionElement& session) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        size_t baseIdx = batch * MODEL_INPUT_SIZE_BACKEND;
        size_t prevBaseIdx = (batch == 0 ? BATCH_SIZE - 1 : batch - 1) * MODEL_INPUT_SIZE_BACKEND;

        for (size_t j = 1; j < MODEL_INPUT_SIZE_BACKEND; j++) {
            session.processedModelInput[baseIdx + j - 1] = session.processedModelInput[prevBaseIdx + j];
        }

        session.processedModelInput[baseIdx + MODEL_INPUT_SIZE_BACKEND - 1] = session.sendBuffer.popSample(0);
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

void InferenceThreadPool::setBackend(InferenceBackend backend, int sessionID) {
    sessions.at(sessionID)->currentBackend.store(backend);
}

ThreadSafeBuffer& InferenceThreadPool::getSendBuffer(int sessionID) {
    return sessions.at(sessionID)->sendBuffer;
}

ThreadSafeBuffer& InferenceThreadPool::getReceiveBuffer(int sessionID) {
    return sessions.at(sessionID)->receiveBuffer;
}


