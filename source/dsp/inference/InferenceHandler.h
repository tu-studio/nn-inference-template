#ifndef NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H

#include <InferenceManager.h>
#include <PrePostProcessor.h>
#include <InferenceConfig.h>

class InferenceHandler {
public:
    InferenceHandler() = delete;
    InferenceHandler(PrePostProcessor &prePostProcessor, InferenceConfig& config);
    ~InferenceHandler() = default;

    void setInferenceBackend(InferenceBackend inferenceBackend);
    InferenceBackend getInferenceBackend();

    void prepare(HostAudioConfig newAudioConfig);
    void process(float ** inputBuffer, const size_t inputSamples); // buffer[channel][index]

    int getLatency();
    InferenceManager &getInferenceManager(); // TODO remove

private:
    InferenceManager inferenceManager;
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEHANDLER_H
