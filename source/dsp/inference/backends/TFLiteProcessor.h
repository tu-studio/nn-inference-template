#ifndef NN_INFERENCE_TEMPLATE_TFLITEPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_TFLITEPROCESSOR_H

#include <JuceHeader.h>
#include "../InferenceBuffer.h"
#include <InferenceConfig.h>
#include "tensorflow/lite/c_api.h"

class TFLiteProcessor {
public:
    TFLiteProcessor(InferenceConfig& config);
    ~TFLiteProcessor();

    void prepareToPlay();
    void processBlock(NNInferenceTemplate::InputArray& input, NNInferenceTemplate::OutputArray& output);

private:
    InferenceConfig& inferenceConfig;

    TfLiteModel* model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter* interpreter;

    TfLiteTensor* inputTensor;
    const TfLiteTensor* outputTensor;
};

#endif //NN_INFERENCE_TEMPLATE_TFLITEPROCESSOR_H
