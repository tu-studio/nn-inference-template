#ifndef NN_INFERENCE_TEMPLATE_HOSTAUDIOCONFIG_H
#define NN_INFERENCE_TEMPLATE_HOSTAUDIOCONFIG_H

#include <cstddef>

struct HostAudioConfig {
    size_t hostChannels;
    size_t hostBufferSize;
    double hostSampleRate;
};

#endif //NN_INFERENCE_TEMPLATE_HOSTAUDIOCONFIG_H
