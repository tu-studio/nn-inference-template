//
// Created by Valentin Ackva on 14/12/2023.
//

#ifndef NN_INFERENCE_TEMPLATE_DETAILVIEWBUTTON_H
#define NN_INFERENCE_TEMPLATE_DETAILVIEWBUTTON_H

#include "JuceHeader.h"
#include "../../PluginProcessor.h"
#include "DetailWindow.h"

class DetailViewButton : public juce::Component {
public:
    DetailViewButton(AudioPluginAudioProcessor& p);
    ~DetailViewButton() override;

private:
    void resized() override;
    void createSecondWindow();

private:
    AudioPluginAudioProcessor& processorRef;
    std::unique_ptr<DetailWindow> window;
    juce::DrawableButton button;
    std::unique_ptr<juce::Drawable> background = juce::Drawable::createFromImageData (BinaryData::detailButtonOn_svg, BinaryData::detailButtonOn_svgSize);
};

#endif //NN_INFERENCE_TEMPLATE_DETAILVIEWBUTTON_H
