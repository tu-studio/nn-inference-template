/* ==========================================================================

Minimal TensorFlow Lite example from https://www.tensorflow.org/lite/guide/inference
Licence: Apache 2.0

========================================================================== */

#include <cstdio>
#include <iostream>
#include <array>
#include "tensorflow/lite/c_api.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

int main(int argc, char* argv[]) {

    std::cout << "Minimal TensorFlow-Lite example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    std::string modelpath = "";
    int batchSize = 0;
    int modelInputSize = 0;
    int modelOutputSize = 0;

#if MODEL_TO_USE == 1
    std::string filepath = GUITARLSTM_MODELS_PATH_TENSORFLOW;
    modelpath = filepath + "model_0/model_0-minimal.tflite";

    batchSize = 2;
    modelInputSize = 150;
    modelOutputSize = 1;
#elif MODEL_TO_USE == 2
    std::string filepath = STEERABLENAFX_MODELS_PATH_TENSORFLOW;
    modelpath = filepath + "model_0/steerable-nafx.tflite";

    batchSize = 1;
    modelInputSize = 56236;
    modelOutputSize = 64;
#endif

    // Load model
    TfLiteModel* model = TfLiteModelCreateFromFile(modelpath.c_str());

    // Create the interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    // Limit inference to one thread
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    // Allocate memory for all tensors
    TfLiteInterpreterAllocateTensors(interpreter);

    // Get input tensor
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    std::cout << "Input shape 0: " << TfLiteTensorDim(inputTensor, 0) << '\n';
    std::cout << "Input shape 1: " << TfLiteTensorDim(inputTensor, 1) << '\n';
    std::cout << "Input shape 2: " << TfLiteTensorDim(inputTensor, 2) << '\n';

    // Fill input tensor with data
    const int inputSize = batchSize * modelInputSize;
    float inputData[inputSize];
    for (int i = 0; i < inputSize; i++) {
        inputData[i] = i * 0.001f;
    }
    TfLiteTensorCopyFromBuffer(inputTensor, &inputData, inputSize * sizeof(float));

    // Execute inference.
    TfLiteInterpreterInvoke(interpreter);

    // Get output tensor
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    // Extract the output tensor data
    const int outputSize = batchSize * modelOutputSize;
    float outputData[outputSize];
    TfLiteTensorCopyToBuffer(outputTensor, &outputData, outputSize * sizeof(float));

    std::cout << "Output shape 0: " << TfLiteTensorDim(outputTensor, 0) << '\n';
    std::cout << "Output shape 1: " << TfLiteTensorDim(outputTensor, 1) << '\n';
    std::cout << "Output shape 2: " << TfLiteTensorDim(outputTensor, 2) << '\n';

    for (int i = 0; i < outputSize; i++) {
        std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
    }

    // Dispose of the model and interpreter objects.
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    
    return 0;
}
