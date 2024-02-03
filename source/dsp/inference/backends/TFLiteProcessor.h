#ifndef NN_INFERENCE_TEMPLATE_TFLITEPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_TFLITEPROCESSOR_H


#include <InferenceConfig.h>
#include "../utils/AudioBuffer.h"
#include <tensorflow/lite/c_api.h>

class TFLiteProcessor {
public:
    TFLiteProcessor(InferenceConfig& config);
    ~TFLiteProcessor();

    void prepareToPlay();
    void processBlock(AudioBufferF& input, AudioBufferF& output);

private:
    InferenceConfig& inferenceConfig;

    TfLiteModel* model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter* interpreter;

    TfLiteTensor* inputTensor;
    const TfLiteTensor* outputTensor;
};

#endif //NN_INFERENCE_TEMPLATE_TFLITEPROCESSOR_H
