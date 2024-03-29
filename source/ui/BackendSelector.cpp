#include "BackendSelector.h"

BackendSelector::BackendSelector(juce::AudioProcessorValueTreeState& state) : apvts(state) {
    backendList = PluginParameters::backendTypes;
    currentBackend = stringToBackend(PluginParameters::defaultBackend);
}

void BackendSelector::setBackend(int backendID) {
    juce::String type = backendList.getReference(backendID);
    auto newBackend = stringToBackend(type);

    if (newBackend != currentBackend) {
        currentBackend = newBackend;
        repaint();
    }
}

anira::InferenceBackend BackendSelector::getBackend() {
    return currentBackend;
}

void BackendSelector::paint(juce::Graphics &g) {
    auto currentBound = getBounds();
    switch (currentBackend) {
#ifdef USE_TFLITE
        case anira::InferenceBackend::TFLITE:
            backendTFLite->drawWithin(g, currentBound.toFloat(), juce::RectanglePlacement::stretchToFit, 1.0f);
            break;
#endif
#ifdef USE_LIBTORCH
        case anira::InferenceBackend::LIBTORCH:
            backendLibTorch->drawWithin(g, currentBound.toFloat(), juce::RectanglePlacement::stretchToFit, 1.0f);
            break;
#endif
#ifdef USE_ONNXRUNTIME
        case anira::InferenceBackend::ONNX:
            backendONNX->drawWithin(g, currentBound.toFloat(), juce::RectanglePlacement::stretchToFit, 1.0f);
            break;
#endif
    }

    switch (currentHover) {
        case TFLITE_HOVER:
            highlight->drawWithin(g, tfliteHighlightBounds.toFloat(), juce::RectanglePlacement::stretchToFit, 1.0f);
            break;
        case LIBTORCH_HOVER:
            highlight->drawWithin(g, libtorchHighlightBounds.toFloat(), juce::RectanglePlacement::stretchToFit, 1.0f);
            break;
        case ONNX_HOVER:
            highlight->drawWithin(g, onnxHighlightBounds.toFloat(), juce::RectanglePlacement::stretchToFit, 1.0f);
            break;
        case NONE:
            break;
    }

    // Debug: Show hover hit boxes
    // g.setColour(juce::Colours::red);
    // g.drawRect(tfliteBounds);
    // g.drawRect(libtorchBounds);
    // g.drawRect(onnxBounds);
}

void BackendSelector::resized() {
    tfliteBounds = juce::Rectangle<int>(90, 280, 340, 70);
    libtorchBounds = juce::Rectangle<int>(90, 355, 340, 50);
    onnxBounds = juce::Rectangle<int>(90, 411, 340, 55);

    tfliteHighlightBounds = juce::Rectangle<int>(0, 205, getWidth(), 150);
    libtorchHighlightBounds = juce::Rectangle<int>(0, 258, getWidth(), 150);
    onnxHighlightBounds = juce::Rectangle<int>(0, 320, getWidth(), 150);
}

void BackendSelector::mouseExit(const juce::MouseEvent &) {
    if (currentHover != NONE) {
        currentHover = NONE;
        repaint();
    }
}

void BackendSelector::mouseMove(const juce::MouseEvent &event) {
    auto pos = event.getPosition();

    if ( libtorchBounds.contains(pos) ) {
        currentHover = LIBTORCH_HOVER;
    } else if ( tfliteBounds.contains(pos) ) {
        currentHover = TFLITE_HOVER;
    } else if ( onnxBounds.contains(pos) ) {
        currentHover = ONNX_HOVER;
    } else if (currentHover != NONE) {
        currentHover = NONE;
    } else {
        return;
    }

    repaint();
}

void BackendSelector::mouseDown(const juce::MouseEvent &event) {
    auto pos = event.getPosition();
    if ( libtorchBounds.contains(pos) ) {
#ifdef USE_LIBTORCH
        currentBackend = anira::InferenceBackend::LIBTORCH;
#endif
    } else if ( tfliteBounds.contains(pos) ) {
#ifdef USE_TFLITE
        currentBackend = anira::InferenceBackend::TFLITE;
#endif
    } else if ( onnxBounds.contains(pos) ) {
#ifdef USE_ONNXRUNTIME
        currentBackend = anira::InferenceBackend::ONNX;
#endif
    } else {
        getNextBackend();
    }

    backendChanged();

    repaint();
}

void BackendSelector::backendChanged() {
    juce::AudioParameterChoice* choice = nullptr;
    choice = dynamic_cast<juce::AudioParameterChoice*>(apvts.getParameter (PluginParameters::BACKEND_TYPE_ID.getParamID()));
    *choice = getCurrentBackendID();
}

int BackendSelector::getCurrentBackendID() {
    return backendList.indexOf(backendToString(currentBackend));
}

void BackendSelector::getNextBackend() {
//TODO
//    switch (currentBackend) {
//#ifdef USE_TFLITE
//        case anira::InferenceBackend::TFLITE:
//            currentBackend = anira::InferenceBackend::LIBTORCH;
//            break;
//#endif
//#ifdef USE_LIBTORCH
//        case anira::InferenceBackend::LIBTORCH:
//            currentBackend = anira::InferenceBackend::ONNX;
//            break;
//#endif
//#ifdef USE_ONNXRUNTIME
//        case anira::InferenceBackend::ONNX:
//            currentBackend = anira::InferenceBackend::TFLITE;
//            break;
//#endif
//    }
}

anira::InferenceBackend BackendSelector::stringToBackend(juce::String &backendStr) {
#ifdef USE_TFLITE
    if (backendStr == "TFLITE") return anira::InferenceBackend::TFLITE;
#endif
#ifdef USE_LIBTORCH
    if (backendStr == "LIBTORCH") return anira::InferenceBackend::LIBTORCH;
#endif
#ifdef USE_ONNXRUNTIME
    if (backendStr == "ONNXRUNTIME") return anira::InferenceBackend::ONNX;
#endif
    throw std::invalid_argument("Invalid backend string");
}

juce::String BackendSelector::backendToString(anira::InferenceBackend backend) {
    switch (backend) {
#ifdef USE_TFLITE
        case anira::InferenceBackend::TFLITE:
            return "TFLITE";
#endif
#ifdef USE_LIBTORCH
        case anira::InferenceBackend::LIBTORCH:
            return "LIBTORCH";
#endif
#ifdef USE_ONNXRUNTIME
        case anira::InferenceBackend::ONNX:
            return "ONNXRUNTIME";
#endif
        default:
            return "";
    }
}
