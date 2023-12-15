//
// Created by Valentin Ackva on 14/12/2023.
//

#ifndef NN_INFERENCE_TEMPLATE_DETAILWINDOWCOMPONENT_H
#define NN_INFERENCE_TEMPLATE_DETAILWINDOWCOMPONENT_H

#include "JuceHeader.h"
#include "../../PluginProcessor.h"

class DetailWindowComponent : public juce::Component, public juce::Timer {
public:
    DetailWindowComponent(AudioPluginAudioProcessor& p);
    ~DetailWindowComponent() override;

private:
    void timerCallback() override;
    void resized() override;

private:
    AudioPluginAudioProcessor& processorRef;
    juce::Label missingBlocksTitle, currentSessionIDTitle, numOfSessionsTitle, missingBlocksValue, currentSessionIDValue, numOfSessionsValue;

    std::vector<juce::Label*> titleLabels {&missingBlocksTitle, &currentSessionIDTitle, &numOfSessionsTitle};
    std::vector<juce::Label*> valueLabels {&missingBlocksValue, &currentSessionIDValue, &numOfSessionsValue};

    int missingBlocks, numOfSessions;
};


#endif //NN_INFERENCE_TEMPLATE_DETAILWINDOWCOMPONENT_H
