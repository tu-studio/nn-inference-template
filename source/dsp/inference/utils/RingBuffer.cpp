#include "RingBuffer.h"

RingBuffer::RingBuffer() = default;

void RingBuffer::initialise(size_t numChannels, size_t numSamples) {
    readPos.resize(numChannels);
    writePos.resize(numChannels);

    for (size_t i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }

    nChannels = numChannels;
    nSamples = numSamples;

    allocateVector();
}

void RingBuffer::reset() {
    resetVector();

    for (size_t i = 0; i < readPos.size(); i++) {
        readPos[i] = 0;
        writePos[i] = 0;
    }
}

void RingBuffer::pushSample(float sample, size_t channel) {
    if (std::isnan(sample)){
        sample = 0.f;
    }
    buffer[channel][writePos[channel]] = sample;

    ++writePos[channel];

    if (writePos[channel] >= nSamples) {
        writePos[channel] = 0;
    }
}

float RingBuffer::popSample(size_t channel) {
    auto sample = buffer[channel][readPos[channel]];

    ++readPos[channel];

    if (readPos[channel] >= nSamples) {
        readPos[channel] = 0;
    }

    if (std::isnan(sample)){
        return 0.f;
    } else {
        return sample;
    }
}

float RingBuffer::getSample (size_t channel, size_t offset) {
    if ((int) readPos[channel] - (int) offset < 0) {
        return buffer[channel][nSamples + readPos[channel] - offset];
    } else {
        return buffer[channel][readPos[channel] - offset];
    }
}

size_t RingBuffer::getAvailableSamples(size_t channel) {
    size_t returnValue;

    if (readPos[channel] <= writePos[channel]) {
        returnValue = writePos[channel] - readPos[channel];
    } else {
        returnValue = writePos[channel] + nSamples - readPos[channel];
    }

    return returnValue;
}

void RingBuffer::allocateVector() {
    float initialValue = 0.0f;

    buffer.resize(nChannels);

    for (auto& innerBuffer : buffer) {
        innerBuffer.resize(nSamples, initialValue);
    }
}

void RingBuffer::resetVector() {
    float newValue = 0.0f;

    for (auto& innerBuffer : buffer) {
        std::fill(innerBuffer.begin(), innerBuffer.end(), newValue);
    }
}