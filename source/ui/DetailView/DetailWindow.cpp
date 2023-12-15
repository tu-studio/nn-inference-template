//
// Created by Valentin Ackva on 14/12/2023.
//

#include "DetailWindow.h"

DetailWindow::DetailWindow(AudioPluginAudioProcessor &p) :
        juce::DocumentWindow("Details", juce::Colours::darkgrey, 4),
        detailsWindowComponent(p)
{
    setSize(600, 400);
    setContentOwned(&detailsWindowComponent, true);
}

void DetailWindow::closeButtonPressed() {
    this->setVisible(false);
}
