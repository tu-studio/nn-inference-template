#ifndef VAESYNTH_RINGBUFFER_H
#define VAESYNTH_RINGBUFFER_H

#include <JuceHeader.h>

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
    juce::AudioBuffer<float> buffer;
    std::vector<size_t> readPos, writePos;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RingBuffer)
};

#endif //VAESYNTH_RINGBUFFER_H