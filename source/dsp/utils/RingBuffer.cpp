#include "RingBuffer.h"

RingBuffer::RingBuffer() = default;

void RingBuffer::initialise(int numChannels, int numSamples) {
    readPos.resize(numChannels);
    writePos.resize(numChannels);

    for (int i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }

    buffer.setSize(numChannels, numSamples);
}

void RingBuffer::reset() {
    buffer.clear();
    for (int i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }
}

void RingBuffer::pushSample(float sample, int channel) {
    if (std::isnan(sample)){
        sample = 0.f;
//        std::cout << "Sample is nan! push" << std::endl; //DBG
    }
    buffer.setSample(channel, writePos[channel], sample);

    ++writePos[channel];

    if (writePos[channel] >= buffer.getNumSamples()) {
        writePos[channel] = 0;
    }
}

float RingBuffer::popSample(int channel) {
    auto sample = buffer.getSample(channel, readPos[channel]);

    ++readPos[channel];

    if (readPos[channel] >= buffer.getNumSamples()) {
        readPos[channel] = 0;
    }
    if (std::isnan(sample)){
//        std::cout << "Sample is nan! pop" << std::endl; //DBG
        return 0.f;
    }
    else return sample;
}

float RingBuffer::getSample (int channel, unsigned int offset) {
    if (readPos[channel] - offset < 0) {
        return buffer.getSample(channel, buffer.getNumSamples() + readPos[channel] - offset);
    } else {
        return buffer.getSample(channel, readPos[channel] - offset);
    }
}

int RingBuffer::getAvailableSamples(int channel) {
    int returnValue;

    if (readPos[channel] <= writePos[channel]) {
        returnValue = writePos[channel] - readPos[channel];
    } else {
        returnValue = writePos[channel] + buffer.getNumSamples() - readPos[channel];
    }

    return returnValue;
}