#include "TFLiteProcessor.h"

#ifdef _WIN32
#include <comdef.h>
#endif

TFLiteProcessor::TFLiteProcessor()
{
#ifdef _WIN32
    _bstr_t modelPathChar (modelpath.c_str());
    model = TfLiteModelCreateFromFile(modelPathChar);
#else
    model = TfLiteModelCreateFromFile(modelpath.c_str());
#endif

    options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    interpreter = TfLiteInterpreterCreate(model, options);
}

TFLiteProcessor::~TFLiteProcessor()
{
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}

void TFLiteProcessor::prepareToPlay() {
    TfLiteInterpreterAllocateTensors(interpreter);
    inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    if (WARM_UP) {
        AudioBufferF input(1, BATCH_SIZE * MODEL_INPUT_SIZE_BACKEND);
        AudioBufferF output(1, BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND);
        processBlock(input, output);
    }
}

void TFLiteProcessor::processBlock(AudioBufferF& input, AudioBufferF& output) {
    TfLiteTensorCopyFromBuffer(inputTensor, input.getRawData(), input.getNumSamples() * sizeof(float)); //TODO: Multichannel support
    TfLiteInterpreterInvoke(interpreter);
    TfLiteTensorCopyToBuffer(outputTensor, output.getRawData(), output.getNumChannels() * sizeof(float)); //TODO: Multichannel support
}