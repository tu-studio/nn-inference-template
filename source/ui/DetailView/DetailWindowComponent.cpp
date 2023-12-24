//
// Created by Valentin Ackva on 14/12/2023.
//

#include "DetailWindowComponent.h"

DetailWindowComponent::DetailWindowComponent(AudioPluginAudioProcessor &p) :
        processorRef(p)
{
    setSize(600, 400);
    missingBlocksTitle.setJustificationType(juce::Justification::right);
    missingBlocksTitle.setText("Missing Blocks:", juce::dontSendNotification);
    addAndMakeVisible(missingBlocksTitle);

    currentSessionIDTitle.setJustificationType(juce::Justification::right);
    currentSessionIDTitle.setText("Sessions ID:", juce::dontSendNotification);
    addAndMakeVisible(currentSessionIDTitle);

    numOfSessionsTitle.setJustificationType(juce::Justification::right);
    numOfSessionsTitle.setText("Active Sessions:", juce::dontSendNotification);
    addAndMakeVisible(numOfSessionsTitle);

    missingBlocksValue.setJustificationType(juce::Justification::left);
    missingBlocksValue.setText("-", juce::dontSendNotification);
    addAndMakeVisible(missingBlocksValue);

    currentSessionIDValue.setJustificationType(juce::Justification::left);
    const int sessionID = processorRef.getInferenceManager().getSessionID();
    currentSessionIDValue.setText(juce::String{sessionID}, juce::dontSendNotification);
    addAndMakeVisible(currentSessionIDValue);

    numOfSessionsValue.setJustificationType(juce::Justification::left);
    numOfSessionsValue.setText("-", juce::dontSendNotification);
    addAndMakeVisible(numOfSessionsValue);

    startTimerHz(30);
}

DetailWindowComponent::~DetailWindowComponent() {
    stopTimer();
}

void DetailWindowComponent::timerCallback() {
    if (!isVisible()) return;

    auto newMissingBlocks = processorRef.getInferenceManager().getMissingBlocks();
    // TODO access through class InferenceThread itself
    auto newNumOfSessions = processorRef.getInferenceManager().getInferenceThread().getNumberOfSessions();

    if (newMissingBlocks != missingBlocks) {
        missingBlocks = newMissingBlocks;
        juce::String missingBlockValueString {missingBlocks};
        missingBlocksValue.setText(missingBlockValueString, juce::dontSendNotification);
    }

    if (newNumOfSessions != numOfSessions) {
        numOfSessions = newNumOfSessions;
        juce::String missingBlockValueString {numOfSessions};
        numOfSessionsValue.setText(missingBlockValueString, juce::dontSendNotification);
    }
}

void DetailWindowComponent::resized() {
    const int gab = 10;
    auto titleBounds = getBounds().removeFromLeft(getWidth() / 3);
    auto valueBounds = getBounds().removeFromRight(getWidth() - gab - titleBounds.getWidth());

    auto heightPerElement = getHeight() / (int) titleLabels.size();

    for (int i = 0; i < (int) titleLabels.size(); ++i) {
        auto titleBound = titleBounds.removeFromTop(heightPerElement);
        auto valueBound = valueBounds.removeFromTop(heightPerElement);

        titleLabels[(size_t) i]->setBounds(titleBound);
        valueLabels[(size_t) i]->setBounds(valueBound);
    }
}
