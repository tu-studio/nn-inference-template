//
// Created by Valentin Ackva on 08/12/2023.
//

#ifndef NN_INFERENCE_TEMPLATE_AUDIOCONFIG_H
#define NN_INFERENCE_TEMPLATE_AUDIOCONFIG_H

struct HostConfig {
    int hostChannels;
    int hostBufferSize;
    double hostSampleRate;
};

#endif //NN_INFERENCE_TEMPLATE_AUDIOCONFIG_H
