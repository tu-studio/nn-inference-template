//
// Created by Valentin Ackva on 14/12/2023.
//

#include "DetailViewButton.h"

DetailViewButton::DetailViewButton(AudioPluginAudioProcessor &p) : processorRef(p), button("detailView", juce::DrawableButton::ButtonStyle::ImageStretched) {
    addAndMakeVisible(button);
    button.setImages(background.get());

    button.onClick = [this]() {
        createSecondWindow();
    };
}

DetailViewButton::~DetailViewButton() {
    window.reset();
}

void DetailViewButton::resized() {
    button.setBounds(getLocalBounds());
}

void DetailViewButton::createSecondWindow() {
    if (window) {
        window->setVisible(!window->isVisible());
    } else {
        window = std::make_unique<DetailWindow>(processorRef);
        window->addToDesktop ();
        window->centreWithSize (600, 400);
        window->setVisible (true);
    }
}

