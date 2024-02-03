#ifndef NN_INFERENCE_TEMPLATE_INFERENCEBUFFER_H
#define NN_INFERENCE_TEMPLATE_INFERENCEBUFFER_H

#include <array>

// TODO replace this class with AudioBuffer

#if MODEL_TO_USE == 1

#define BATCH_SIZE 128
#define MODEL_INPUT_SIZE_BACKEND 150 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_OUTPUT_SIZE_BACKEND 1

#elif MODEL_TO_USE == 2

#define BATCH_SIZE 1
#define MODEL_INPUT_SIZE_BACKEND 15380 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_OUTPUT_SIZE_BACKEND 2048

#elif MODEL_TO_USE == 3

#define BATCH_SIZE 1
#define MODEL_INPUT_SIZE_BACKEND 2048 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_OUTPUT_SIZE_BACKEND 2048

#endif // MODEL_TO_USE

namespace NNInferenceTemplate {
    using InputArray = std::array<float, BATCH_SIZE * MODEL_INPUT_SIZE_BACKEND>;
    using OutputArray = std::array<float, BATCH_SIZE * MODEL_OUTPUT_SIZE_BACKEND>;
}

#endif //NN_INFERENCE_TEMPLATE_INFERENCEBUFFER_H