//
// Created by Valentin Ackva on 02/02/2024.
//

#ifndef NN_INFERENCE_TEMPLATE_INFERENCEBACKEND_H
#define NN_INFERENCE_TEMPLATE_INFERENCEBACKEND_H

enum InferenceBackend {
#ifdef USE_LIBTORCH
    LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
    ONNX,
#endif
#ifdef USE_TFLITE
    TFLITE,
#endif
};

#endif //NN_INFERENCE_TEMPLATE_INFERENCEBACKEND_H
