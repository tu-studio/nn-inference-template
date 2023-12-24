#include "RingBuffer.h"

RingBuffer::RingBuffer() = default;

void RingBuffer::initialise(size_t numChannels, size_t numSamples) {
    readPos.resize(numChannels);
    writePos.resize(numChannels);

    for (size_t i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }

    buffer.setSize(numChannels, numSamples);
}

void RingBuffer::reset() {
    buffer.clear();
    for (size_t i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }
}

void RingBuffer::pushSample(float sample, size_t channel) {
    if (std::isnan(sample)){
        sample = 0.f;
//        std::cout << "Sample is nan! push" << std::endl; //DBG
    }
    buffer.setSample(channel, writePos[channel], sample);

    ++writePos[channel];

    if (writePos[channel] >= (size_t) buffer.getNumSamples()) {
        writePos[channel] = 0;
    }
}

float RingBuffer::popSample(size_t channel) {
    auto sample = buffer.getSample(channel, readPos[channel]);

    ++readPos[channel];

    if (readPos[channel] >= (size_t) buffer.getNumSamples()) {
        readPos[channel] = 0;
    }
    if (std::isnan(sample)){
//        std::cout << "Sample is nan! pop" << std::endl; //DBG
        return 0.f;
    }
    else return sample;
}

float RingBuffer::getSample (size_t channel, size_t offset) {
    if ((int) readPos[channel] - (int) offset < 0) {
        return buffer.getSample(channel, (size_t) buffer.getNumSamples() + readPos[channel] - offset);
    } else {
        return buffer.getSample(channel, readPos[channel] - offset);
    }
}

size_t RingBuffer::getAvailableSamples(size_t channel) {
    size_t returnValue;

    if (readPos[channel] <= writePos[channel]) {
        returnValue = writePos[channel] - readPos[channel];
    } else {
        returnValue = writePos[channel] + (size_t) buffer.getNumSamples() - readPos[channel];
    }

    return returnValue;
}