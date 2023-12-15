//
// Created by Valentin Ackva on 14/12/2023.
//

#ifndef NN_INFERENCE_TEMPLATE_DETAILWINDOW_H
#define NN_INFERENCE_TEMPLATE_DETAILWINDOW_H

#include "JuceHeader.h"
#include "../../PluginProcessor.h"
#include "DetailWindowComponent.h"

class DetailWindow : public juce::DocumentWindow {
public:
    DetailWindow(AudioPluginAudioProcessor& p);

private:
    void closeButtonPressed() override;

private:
    DetailWindowComponent detailsWindowComponent;
};



#endif //NN_INFERENCE_TEMPLATE_DETAILWINDOW_H
