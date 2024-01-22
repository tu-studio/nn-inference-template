#include "LibtorchProcessor.h"

LibtorchProcessor::LibtorchProcessor() {
#if WIN32
    _putenv("OMP_NUM_THREADS=1");
    _putenv("MKL_NUM_THREADS=1");
#else
    const char* ompNumThreads = "OMP_NUM_THREADS=1";
    const char* mklNumThreads = "MKL_NUM_THREADS=1";
    putenv(const_cast<char*>(ompNumThreads));
    putenv(const_cast<char*>(mklNumThreads));
#endif

    try {
        module = torch::jit::load(filepath + modelname);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cout << e.what() << std::endl;
    }
}

LibtorchProcessor::~LibtorchProcessor() {
}

void LibtorchProcessor::prepareToPlay() {
    inputs.clear();
    inputs.push_back(torch::zeros(MODEL_INPUT_SHAPE_LIBTORCH));

    if (WARM_UP) {
        NNInferenceTemplate::InputArray input;
        NNInferenceTemplate::OutputArray output;
        processBlock(input, output);
    }
}

void LibtorchProcessor::processBlock(NNInferenceTemplate::InputArray& input, NNInferenceTemplate::OutputArray& output) {
    // Create input tensor object from input data values and shape
    inputTensor = torch::from_blob(input.data(), (const long long) input.size()).reshape(MODEL_INPUT_SHAPE_LIBTORCH);

    inputs[0] = inputTensor;

    // Run inference
    outputTensor = module.forward(inputs).toTensor();

    // Extract the output tensor data
    for (size_t i = 0; i < BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND; i++) {
#if MODEL_TO_USE == 1
        output[i] = outputTensor[(int64_t) i][0].item<float>();
#elif MODEL_TO_USE == 2
        output[i] = outputTensor[0][0][(int64_t) i].item<float>();
#elif MODEL_TO_USE == 3
        output[i] = outputTensor[(int64_t) i][0][0].item<float>();
#endif
    }
}