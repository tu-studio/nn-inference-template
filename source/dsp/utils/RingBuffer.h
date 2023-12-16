#ifndef VAESYNTH_RINGBUFFER_H
#define VAESYNTH_RINGBUFFER_H

#include <JuceHeader.h>

class RingBuffer
{
public:
    RingBuffer();

    void initialise(int numChannels, int numSamples);
    void reset();
    void pushSample(float sample, int channel);
    float popSample(int channel);
    float getSample(int channel, unsigned int offset);
    int getAvailableSamples(int channel);

private:
    juce::AudioBuffer<float> buffer;
    std::vector<int> readPos, writePos;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RingBuffer)
};
#endif //VAESYNTH_RINGBUFFER_H