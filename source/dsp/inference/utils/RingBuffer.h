#ifndef VAESYNTH_RINGBUFFER_H
#define VAESYNTH_RINGBUFFER_H

#include <vector>
#include <cmath>

class RingBuffer
{
public:
    RingBuffer();

    void initialise(size_t numChannels, size_t numSamples);
    void reset();
    void pushSample(float sample, size_t channel);
    float popSample(size_t channel);
    float getSample(size_t channel, size_t offset);
    size_t getAvailableSamples(size_t channel);

private:
    void allocateVector();
    void resetVector();

private:
    std::vector<std::vector<float>> buffer;
    std::vector<size_t> readPos, writePos;

    size_t nChannels;
    size_t nSamples;
};

#endif //VAESYNTH_RINGBUFFER_H